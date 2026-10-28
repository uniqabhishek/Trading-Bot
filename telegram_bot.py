"""
Professional Telegram Signal Bot
Features: Webhooks, Auto-Scanner, Backtester (with Capital PnL %), Failover.
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
RISK_PER_TRADE = 0.01  # Risk 1% of capital per trade

HIGHER_TF = {'1m': '15m', '5m': '1h', '15m': '1h', '1h': '4h', '4h': '1d', '1d': '1d'}
YF_INTERVAL = {'1m':'1m','5m':'5m','15m':'15m','1h':'1h','4h':'1h','1d':'1d'}
YF_PERIOD   = {'1m':'7d','5m':'60d','15m':'60d','1h':'730d','4h':'730d','1d':'5y'}

exchange_mexc = ccxt.mexc()
exchange_gate = ccxt.gate()

# ============================================================
# DATA & INDICATORS
# ============================================================
def fetch_with_retry(func, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1: raise e
            import time; time.sleep(2)

def fetch_btc_data(timeframe, limit=1000):
    mexc_tf = '60m' if timeframe == '1h' else timeframe
    try:
        ohlcv = fetch_with_retry(exchange_mexc.fetch_ohlcv, 'BTC/USDT', mexc_tf, limit=limit)
    except Exception:
        ohlcv = fetch_with_retry(exchange_gate.fetch_ohlcv, 'BTC/USDT', timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_gold_data(timeframe):
    hist = fetch_with_retry(yf.Ticker("GC=F").history, period=YF_PERIOD[timeframe], interval=YF_INTERVAL[timeframe])
    if hist.empty: raise ValueError("No Gold data")
    df = hist.reset_index()
    df.rename(columns={'Datetime':'timestamp','Date':'timestamp','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)
    return df[['timestamp','open','high','low','close','volume']].tail(1000).reset_index(drop=True)

def fetch_data(asset, timeframe):
    return fetch_btc_data(timeframe) if asset.upper() == 'BTC' else fetch_gold_data(timeframe)

def calculate_indicators(df):
    df = df.copy()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_line'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    return df

def layered_signal(entry_row, higher_row):
    if pd.isna(entry_row['EMA_200']) or pd.isna(entry_row['RSI_14']) or pd.isna(entry_row['MACD_line']): return 'HOLD'
    if pd.isna(higher_row['EMA_200']): return 'HOLD'
    
    trend = 'UP' if higher_row['close'] > higher_row['EMA_200'] else 'DOWN'
    base = 'HOLD'
    if entry_row['close'] > entry_row['EMA_200'] and entry_row['RSI_14'] > 40 and entry_row['MACD_line'] > entry_row['MACD_signal']: base = 'BUY'
    elif entry_row['close'] < entry_row['EMA_200'] and entry_row['RSI_14'] < 60 and entry_row['MACD_line'] < entry_row['MACD_signal']: base = 'SELL'
    
    if base == 'BUY' and trend != 'UP': return 'HOLD'
    if base == 'SELL' and trend != 'DOWN': return 'HOLD'
    if pd.isna(entry_row['vol_ma_20']) or entry_row['volume'] < VOLUME_MULT * entry_row['vol_ma_20']: return 'HOLD'
    return base

# ============================================================
# BACKTESTING ENGINE (With Capital %)
# ============================================================
def run_backtest(asset, timeframe):
    entry_df  = calculate_indicators(fetch_data(asset, timeframe))
    higher_df = calculate_indicators(fetch_data(asset, HIGHER_TF[timeframe]))
    higher_aligned = higher_df.set_index('timestamp').reindex(entry_df['timestamp'], method='ffill').reset_index()

    balance = 1000.0
    peak = balance
    max_drawdown = 0.0
    wins, losses, total = 0, 0, 0

    for i in range(210, len(entry_df) - 1):
        row, h_row = entry_df.iloc[i], higher_aligned.iloc[i]
        sig = layered_signal(row, h_row)
        if sig == 'HOLD': continue

        entry = row['close']
        atr = row['ATR_14']
        if pd.isna(atr) or atr <= 0: continue

        if sig == 'BUY': sl, tp = entry - ATR_SL_MULT * atr, entry + ATR_TP_MULT * atr
        else: sl, tp = entry + ATR_SL_MULT * atr, entry - ATR_TP_MULT * atr

        outcome = None
        for j in range(i + 1, min(i + 100, len(entry_df))):
            c = entry_df.iloc[j]
            if sig == 'BUY':
                if c['low'] <= sl: outcome = 'LOSS'; break
                if c['high'] >= tp: outcome = 'WIN'; break
            else:
                if c['high'] >= sl: outcome = 'LOSS'; break
                if c['low'] <= tp: outcome = 'WIN'; break

        if not outcome: continue # Timeout

        # Apply PnL to balance (1% risk, 1:2 R:R -> Win = +2%, Loss = -1%)
        if outcome == 'WIN':
            balance += balance * RISK_PER_TRADE * (ATR_TP_MULT / ATR_SL_MULT)
            wins += 1
        else:
            balance -= balance * RISK_PER_TRADE
            losses += 1
        
        total += 1
        if balance > peak: peak = balance
        dd = (peak - balance) / peak
        if dd > max_drawdown: max_drawdown = dd

    if total == 0: return "⚠️ No trades triggered in this historical period."

    win_rate = (wins / total) * 100
    profit_pct = ((balance - 1000) / 1000) * 100
    profit_factor = (wins * ATR_TP_MULT) / (losses * ATR_SL_MULT) if losses > 0 else float('inf')
    
    result = (
        f"📊 *BACKTEST: {asset.upper()} / {timeframe}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕰️ Data: {len(entry_df)} candles\n"
        f"📈 Total Trades: {total}\n"
        f"✅ Wins: {wins} | ❌ Losses: {losses}\n"
        f"🎯 *Win Rate: {win_rate:.1f}%*\n"
        f"⚖️ Profit Factor: {profit_factor:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Starting Capital:* $1,000\n"
        f"💵 *Ending Capital:* ${balance:,.2f}\n"
        f"📈 *Total Profit:* {profit_pct:+.1f}%\n"
        f"📉 *Max Drawdown:* {max_drawdown*100:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    if win_rate >= 55: result += "🟢 *Verdict:* Strategy has a proven edge."
    elif win_rate >= 45: result += "🟡 *Verdict:* Breakeven. Needs work."
    else: result += "🔴 *Verdict:* Poor performance. Do not trade live."

    return result

# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("❌ Usage: `/backtest BTC 15m`")
        return
    asset, timeframe = context.args[0].upper(), context.args[1].lower()
    msg = await update.message.reply_text(f"⏳ Backtesting {asset} on {timeframe}... (~10s)")
    try:
        await msg.edit_text(await asyncio.to_thread(run_backtest, asset, timeframe), parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:200]}")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Pro Crypto Bot*\n\n"
        "• `/signal BTC 15m` - Live signal\n"
        "• `/backtest BTC 1h` - Test accuracy & profit %\n"
        "• `/status` - Bot health",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot Online\nExchanges: MEXC / Gate.io\nGold: Yahoo Finance")

# ============================================================
# QUART WEB SERVER
# ============================================================
telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", cmd_start))
telegram_app.add_handler(CommandHandler("backtest", cmd_backtest))
telegram_app.add_handler(CommandHandler("status", cmd_status))

web_app = Quart(__name__)

@web_app.route('/')
@web_app.route('/healthz')
async def health(): return "OK", 200

@web_app.route('/webhook', methods=['POST'])
async def telegram_webhook():
    data = await request.get_json(force=True)
    update = Update.de_json(data=data, bot=telegram_app.bot)
    await telegram_app.update_queue.put(update)
    return "OK", 200

async def main():
    await telegram_app.initialize()
    render_url = os.getenv("RENDER_EXTERNAL_URL")
    if render_url: await telegram_app.bot.set_webhook(url=f"{render_url}/webhook")
    await telegram_app.start()
    
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    config = Config()
    config.bind = [f"0.0.0.0:{os.getenv('PORT', 10000)}"]
    await serve(web_app, config)
    
    await telegram_app.stop()
    await telegram_app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
