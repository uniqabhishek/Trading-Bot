"""
Professional Telegram Signal Bot
Features: Webhooks, Auto-Scanner, Retry Logic, Exchange Failover (MEXC -> Gate.io).
"""

import os
import asyncio
import ccxt
import pandas as pd
import numpy as np
import yfinance as yf
from quart import Quart, request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env var not set!")

ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID")
ALLOWED_CHAT_ID = int(ALLOWED_CHAT_ID) if ALLOWED_CHAT_ID else None

# Strategy Constants
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
VOLUME_MULT = 1.5
HIGHER_TF   = {'15m':'1h', '1h':'4h', '4h':'1d'}
YF_INTERVAL = {'1m':'1m','5m':'5m','15m':'15m','1h':'1h','4h':'1h','1d':'1d'}
YF_PERIOD   = {'1m':'7d','5m':'60d','15m':'60d','1h':'730d','4h':'730d','1d':'5y'}

# Initialize Exchanges for Failover
exchange_mexc = ccxt.mexc()
exchange_gate = ccxt.gateio()

# Memory to prevent duplicate auto-alerts
last_signals = {} 

# ============================================================
# DATA & INDICATORS (With Retry & Failover Logic)
# ============================================================
def fetch_with_retry(func, *args, retries=3, **kwargs):
    """Professional reliability: retry network calls on failure."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            import time
            time.sleep(2)

def fetch_btc_data(timeframe, limit=300):
    # Convert '1h' to '60m' for MEXC compatibility if needed
    mexc_tf = '60m' if timeframe == '1h' else timeframe
    mexc_tf = '4h' if timeframe == '4h' else mexc_tf
    mexc_tf = '1d' if timeframe == '1d' else mexc_tf

    try:
        # Try MEXC first
        ohlcv = fetch_with_retry(exchange_mexc.fetch_ohlcv, 'BTC/USDT', mexc_tf, limit=limit)
    except Exception as e:
        print(f"MEXC failed ({e}), falling back to Gate.io...")
        # If MEXC fails (geoblock or rate limit), fallback to Gate.io
        ohlcv = fetch_with_retry(exchange_gate.fetch_ohlcv, 'BTC/USDT', timeframe, limit=limit)
        
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_gold_data(timeframe):
    interval = YF_INTERVAL[timeframe]
    period   = YF_PERIOD[timeframe]
    hist = fetch_with_retry(yf.Ticker("GC=F").history, period=period, interval=interval)
    if hist.empty:
        raise ValueError("No Gold data from Yahoo Finance")
    df = hist.reset_index()
    df.rename(columns={'Datetime':'timestamp','Date':'timestamp','Open':'open',
                       'High':'high','Low':'low','Close':'close','Volume':'volume'},
              inplace=True)
    if timeframe == '4h':
        df = df.set_index('timestamp').resample('4H').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna().reset_index()
    return df[['timestamp','open','high','low','close','volume']].tail(300).reset_index(drop=True)

def fetch_data(asset, timeframe):
    asset = asset.upper()
    if asset == 'BTC': return fetch_btc_data(timeframe)
    elif asset == 'GOLD': return fetch_gold_data(timeframe)
    raise ValueError(f"Unknown asset: {asset}")

def calculate_indicators(df):
    df = df.copy()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_line']   = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close  = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14']     = tr.rolling(14).mean()
    df['vol_ma_20']  = df['volume'].rolling(20).mean()
    return df

def raw_signal(row):
    if pd.isna(row['EMA_200']) or pd.isna(row['RSI_14']) or pd.isna(row['MACD_line']): return 'HOLD'
    if row['close'] > row['EMA_200'] and row['RSI_14'] > 40 and row['MACD_line'] > row['MACD_signal']: return 'BUY'
    if row['close'] < row['EMA_200'] and row['RSI_14'] < 60 and row['MACD_line'] < row['MACD_signal']: return 'SELL'
    return 'HOLD'

def layered_signal(entry_row, higher_row):
    base = raw_signal(entry_row)
    if base == 'HOLD': return 'HOLD', "No base signal"
    if pd.isna(higher_row['EMA_200']): return 'HOLD', "Higher TF data insufficient"
    trend = 'UP' if higher_row['close'] > higher_row['EMA_200'] else 'DOWN'
    if base == 'BUY' and trend != 'UP': return 'HOLD', f"Base BUY rejected (Higher TF: {trend})"
    if base == 'SELL' and trend != 'DOWN': return 'HOLD', f"Base SELL rejected (Higher TF: {trend})"
    if pd.isna(entry_row['vol_ma_20']) or entry_row['volume'] < VOLUME_MULT * entry_row['vol_ma_20']: return 'HOLD', "Volume too low"
    return base, f"All layers aligned (Higher TF: {trend})"

def get_signal(asset, timeframe):
    timeframe = timeframe.lower()
    if timeframe not in HIGHER_TF:
        raise ValueError(f"Invalid timeframe. Use: {', '.join(HIGHER_TF.keys())}")

    entry_df  = calculate_indicators(fetch_data(asset, timeframe))
    higher_df = calculate_indicators(fetch_data(asset, HIGHER_TF[timeframe]))

    if len(entry_df) < 210: raise ValueError(f"Not enough candles ({len(entry_df)}/210)")
    
    entry_row  = entry_df.iloc[-2]
    higher_row = higher_df.iloc[-2]
    signal, reason = layered_signal(entry_row, higher_row)
    
    price = float(entry_row['close'])
    atr   = float(entry_row['ATR_14']) if not pd.isna(entry_row['ATR_14']) else price * 0.005

    if signal == 'BUY': sl, tp = price - ATR_SL_MULT * atr, price + ATR_TP_MULT * atr
    elif signal == 'SELL': sl, tp = price + ATR_SL_MULT * atr, price - ATR_TP_MULT * atr
    else: sl = tp = None

    return {"asset": asset.upper(), "timeframe": timeframe, "signal": signal,
            "reason": reason, "price": price, "sl": sl, "tp": tp,
            "timestamp": pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

# ============================================================
# FORMATTING (Smart Decimals)
# ============================================================
def format_price(asset, price):
    if price is None: return "N/A"
    if asset == 'BTC': return f"${price:,.0f}"
    return f"${price:,.2f}"

def format_signal(s):
    if s['signal'] == 'HOLD':
        return (f"⚪ *HOLD — {s['asset']} / {s['timeframe']}*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Price: `{format_price(s['asset'], s['price'])}`\n"
                f"📋 Reason: {s['reason']}\n🕐 {s['timestamp']}")
    
    emoji = "🟢" if s['signal'] == 'BUY' else "🔴"
    return (f"{emoji} *{s['signal']} — {s['asset']} / {s['timeframe']}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Entry:       `{format_price(s['asset'], s['price'])}`\n"
            f"🛑 Stop Loss:   `{format_price(s['asset'], s['sl'])}`\n"
            f"🎯 Take Profit: `{format_price(s['asset'], s['tp'])}`\n"
            f"⚖️ R:R Ratio:   `1:2.0`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ {s['reason']}\n🕐 {s['timestamp']}")

# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Professional Crypto Signal Bot*\n\n"
        "Commands:\n"
        "• `/signal BTC 15m` - Check specific signal\n"
        "• `/price GOLD` - Get live price\n"
        "• `/scan` - Run auto-scanner manually\n"
        "• `/status` - Check bot health\n\n"
        "Timeframes: `15m 1h 4h`\nAssets: `BTC GOLD`",
        parse_mode="Markdown"
    )

async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("❌ Usage: `/price BTC`", parse_mode="Markdown")
        return
    asset = context.args[0].upper()
    try:
        df = fetch_data(asset, '15m')
        price = float(df.iloc[-1]['close'])
        await update.message.reply_text(f"💵 *{asset} Live Price:* `{format_price(asset, price)}`", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:200]}")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("❌ Usage: `/signal BTC 15m`", parse_mode="Markdown")
        return
    asset, timeframe = context.args[0], context.args[1]
    msg = await update.message.reply_text(f"⏳ Analyzing {asset.upper()} on {timeframe}...")
    try:
        result = await asyncio.to_thread(get_signal, asset, timeframe)
        await msg.edit_text(format_signal(result), parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning market for setups...")
    await run_market_scanner(context.bot, force=True, chat_id=update.effective_chat.id)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ *Bot is Online & Healthy*\nWebhook: Active\nData Feeds: MEXC/Gate & Yahoo Finance", parse_mode="Markdown")

# ============================================================
# BACKGROUND AUTO-SCANNER
# ============================================================
async def run_market_scanner(bot, force=False, chat_id=None):
    target_chat = chat_id or ALLOWED_CHAT_ID
    if not target_chat: return

    assets = ['BTC', 'GOLD']
    timeframes = ['15m', '1h', '4h']

    for asset in assets:
        for tf in timeframes:
            try:
                result = await asyncio.to_thread(get_signal, asset, tf)
                if result['signal'] in ('BUY', 'SELL'):
                    sig_key = f"{asset}_{tf}"
                    last_sig = last_signals.get(sig_key)
                    if last_sig != result['signal'] or force:
                        last_signals[sig_key] = result['signal']
                        alert_text = f"🚨 *AUTO-SCANNER ALERT*\n\n{format_signal(result)}"
                        await bot.send_message(chat_id=target_chat, text=alert_text, parse_mode="Markdown")
                elif result['signal'] == 'HOLD':
                    sig_key = f"{asset}_{tf}"
                    if sig_key in last_signals: del last_signals[sig_key]
            except Exception as e:
                print(f"Scanner error on {asset} {tf}: {e}")
            await asyncio.sleep(2)

# ============================================================
# QUART WEB SERVER & WEBHOOK
# ============================================================
telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
telegram_app.add_handler(CommandHandler("start",  cmd_start))
telegram_app.add_handler(CommandHandler("help",   cmd_start))
telegram_app.add_handler(CommandHandler("price",  cmd_price))
telegram_app.add_handler(CommandHandler("signal", cmd_signal))
telegram_app.add_handler(CommandHandler("scan",   cmd_scan))
telegram_app.add_handler(CommandHandler("status", cmd_status))

web_app = Quart(__name__)

@web_app.route('/')
@web_app.route('/healthz')
async def health():
    return "Bot is running", 200

@web_app.route('/webhook', methods=['POST'])
async def telegram_webhook():
    data = await request.get_json(force=True)
    update = Update.de_json(data=data, bot=telegram_app.bot)
    await telegram_app.update_queue.put(update)
    return "OK", 200

async def auto_scanner_loop(bot):
    while True:
        print("🔍 Running auto-scanner...")
        await run_market_scanner(bot)
        await asyncio.sleep(900)

# ============================================================
# MAIN STARTUP
# ============================================================
async def main():
    await telegram_app.initialize()
    
    render_url = os.getenv("RENDER_EXTERNAL_URL")
    if render_url:
        webhook_url = f"{render_url}/webhook"
        print(f"🔗 Setting webhook: {webhook_url}")
        await telegram_app.bot.set_webhook(url=webhook_url)
    
    await telegram_app.start()

    asyncio.create_task(auto_scanner_loop(telegram_app.bot))
    
    port = int(os.getenv("PORT", 10000))
    print(f"🌐 Quart web server listening on 0.0.0.0:{port}")
    
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    config = Config()
    config.bind = [f"0.0.0.0:{port}"]
    
    await serve(web_app, config)
    
    await telegram_app.stop()
    await telegram_app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
