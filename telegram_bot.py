"""
Telegram Signal Bot — Render Webhook Edition
Runs a Web Server (Quart) and receives Telegram messages via Webhooks.
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
# CONFIG (Must be set in Render Environment Variables)
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env var not set!")

ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID")
ALLOWED_CHAT_ID = int(ALLOWED_CHAT_ID) if ALLOWED_CHAT_ID else None

# Strategy constants
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
VOLUME_MULT = 1.5
HIGHER_TF   = {'1m':'15m','5m':'1h','15m':'1h','1h':'4h','4h':'1d','1d':'1d'}
YF_INTERVAL = {'1m':'1m','5m':'5m','15m':'15m','1h':'1h','4h':'1h','1d':'1d'}
YF_PERIOD   = {'1m':'7d','5m':'60d','15m':'60d','1h':'730d','4h':'730d','1d':'5y'}

binance = ccxt.binance()

# ============================================================
# DATA + INDICATORS
# ============================================================
def fetch_btc_data(timeframe, limit=300):
    ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_gold_data(timeframe):
    interval = YF_INTERVAL[timeframe]
    period   = YF_PERIOD[timeframe]
    hist = yf.Ticker("GC=F").history(period=period, interval=interval)
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
    if asset.upper() == 'BTC':
        return fetch_btc_data(timeframe)
    elif asset.upper() == 'GOLD':
        return fetch_gold_data(timeframe)
    raise ValueError(f"Unknown asset: {asset}. Use BTC or GOLD.")

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
    if pd.isna(row['EMA_200']) or pd.isna(row['RSI_14']) or pd.isna(row['MACD_line']):
        return 'HOLD'
    if row['close'] > row['EMA_200'] and row['RSI_14'] > 40 and row['MACD_line'] > row['MACD_signal']:
        return 'BUY'
    if row['close'] < row['EMA_200'] and row['RSI_14'] < 60 and row['MACD_line'] < row['MACD_signal']:
        return 'SELL'
    return 'HOLD'

def layered_signal(entry_row, higher_row):
    base = raw_signal(entry_row)
    if base == 'HOLD': return 'HOLD', "No base signal"
    if pd.isna(higher_row['EMA_200']): return 'HOLD', "Higher TF data insufficient"

    trend = 'UP' if higher_row['close'] > higher_row['EMA_200'] else 'DOWN'
    if base == 'BUY' and trend != 'UP': return 'HOLD', f"Base BUY rejected: higher TF is {trend}"
    if base == 'SELL' and trend != 'DOWN': return 'HOLD', f"Base SELL rejected: higher TF is {trend}"

    if pd.isna(entry_row['vol_ma_20']) or entry_row['volume'] < VOLUME_MULT * entry_row['vol_ma_20']:
        return 'HOLD', "Volume too low (needs 1.5x avg)"
    return base, f"All layers aligned (higher TF: {trend})"

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

    if signal == 'BUY':
        sl, tp = price - ATR_SL_MULT * atr, price + ATR_TP_MULT * atr
    elif signal == 'SELL':
        sl, tp = price + ATR_SL_MULT * atr, price - ATR_TP_MULT * atr
    else:
        sl = tp = None
    return {"asset": asset.upper(), "timeframe": timeframe, "signal": signal,
            "reason": reason, "price": price, "sl": sl, "tp": tp,
            "timestamp": pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

def format_signal(s):
    if s['signal'] == 'HOLD':
        return (f"⚪ *HOLD — {s['asset']} / {s['timeframe']}*\n━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Price: `${s['price']:,.2f}`\n📋 Reason: {s['reason']}\n🕐 {s['timestamp']}")
    emoji = "🟢" if s['signal'] == 'BUY' else "🔴"
    return (f"{emoji} *{s['signal']} — {s['asset']} / {s['timeframe']}*\n━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Entry:       `${s['price']:,.2f}`\n🛑 Stop Loss:   `${s['sl']:,.2f}`\n"
            f"🎯 Take Profit: `${s['tp']:,.2f}`\n⚖️ R:R Ratio:   `1:2.0`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n✅ {s['reason']}\n🕐 {s['timestamp']}")

# ============================================================
# TELEGRAM HANDLERS
# ============================================================
def check_auth(update: Update) -> bool:
    if ALLOWED_CHAT_ID and update.effective_chat.id != ALLOWED_CHAT_ID:
        return False
    return True

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update):
        await update.message.reply_text("🔒 Unauthorized")
        return
    await update.message.reply_text(
        "🤖 *Crypto Signal Bot — 24/7 Webhook Mode*\n\n"
        "Commands:\n• `/signal BTC 15m`\n• `/signal GOLD 1h`\n\n"
        "Timeframes: `1m 5m 15m 1h 4h 1d`\nAssets: `BTC GOLD`",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
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

# ============================================================
# TELEGRAM APP & QUART WEB SERVER
# ============================================================
telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
telegram_app.add_handler(CommandHandler("start",  cmd_start))
telegram_app.add_handler(CommandHandler("help",   cmd_help))
telegram_app.add_handler(CommandHandler("signal", cmd_signal))

web_app = Quart(__name__)

@web_app.route('/')
@web_app.route('/healthz')
async def health():
    return "Bot is running", 200

@web_app.route('/webhook', methods=['POST'])
async def telegram_webhook():
    """Receives updates from Telegram."""
    data = await request.get_json(force=True)
    update = Update.de_json(data=data, bot=telegram_app.bot)
    await telegram_app.update_queue.put(update)
    return "OK", 200

# ============================================================
# STARTUP LOGIC
# ============================================================
async def main():
    # 1. Initialize Telegram App
    await telegram_app.initialize()
    
    # 2. Set Webhook URL using Render's URL environment variable
    render_url = os.getenv("RENDER_EXTERNAL_URL")
    if render_url:
        webhook_url = f"{render_url}/webhook"
        print(f"🔗 Setting webhook to: {webhook_url}")
        await telegram_app.bot.set_webhook(url=webhook_url)
    else:
        print("⚠️ RENDER_EXTERNAL_URL not set. Webhook not configured (Are you running locally?)")
    
    # 3. Start the Telegram app (begins processing updates)
    await telegram_app.start()
    
    # 4. Start Quart server on the port Render expects
    port = int(os.getenv("PORT", 10000))
    print(f"🌐 Quart web server listening on 0.0.0.0:{port}")
    
    # Hypercorn is needed to run Quart in production
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    config = Config()
    config.bind = [f"0.0.0.0:{port}"]
    
    await serve(web_app, config)
    
    # 5. Cleanup on shutdown
    await telegram_app.stop()
    await telegram_app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
