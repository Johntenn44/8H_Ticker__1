import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---

COINS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT",
    "EIGEN/USDT", "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT",
    "DOGE/USDT", "VIRTUAL/USDT", "CAKE/USDT", "GRASS/USDT", "AAVE/USDT",
    "SUI/USDT", "ARB/USDT", "XLM/USDT", "MNT/USDT", "LTC/USDT", "NEAR/USDT",
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # 12-hour candles
LOOKBACK = 210       # Number of candles to fetch (>= 200)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATIONS ---

def add_indicators(df):
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    return df

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# --- TREND ANALYSIS ---

def analyze_rsi_trend(rsi8, rsi13, rsi21):
    if rsi8 > rsi13 > rsi21:
        return "Uptrend"
    elif rsi8 < rsi13 < rsi21:
        return "Downtrend"
    else:
        return "No clear RSI trend"

def analyze_kdj_trend(k_prev, k_curr, d_prev, d_curr, j_curr):
    if k_prev < d_prev and k_curr > d_curr and j_curr > k_curr and j_curr > d_curr:
        return "Bullish KDJ crossover"
    elif k_prev > d_prev and k_curr < d_curr and j_curr < k_curr and j_curr < d_curr:
        return "Bearish KDJ crossover"
    else:
        return "No clear KDJ trend"

# --- DATA FETCHING ---

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, data=payload)
    resp.raise_for_status()

# --- BACKTESTING ---

def backtest_signals(df):
    df = add_indicators(df)
    df['RSI8'] = calculate_rsi(df['close'], 8)
    df['RSI13'] = calculate_rsi(df['close'], 13)
    df['RSI21'] = calculate_rsi(df['close'], 21)
    k, d, j = calculate_kdj(df)
    df['K'] = k
    df['D'] = d
    df['J'] = j

    trades = []
    position = None  # None or dict with entry info

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        timestamp = df.index[i]

        # RSI trend at current candle
        rsi8 = df['RSI8'].iloc[i]
        rsi13 = df['RSI13'].iloc[i]
        rsi21 = df['RSI21'].iloc[i]
        rsi_signal = analyze_rsi_trend(rsi8, rsi13, rsi21)

        # KDJ trend using current and previous candles
        k_prev, k_curr = df['K'].iloc[i-1], df['K'].iloc[i]
        d_prev, d_curr = df['D'].iloc[i-1], df['D'].iloc[i]
        j_curr = df['J'].iloc[i]
        kdj_signal = analyze_kdj_trend(k_prev, k_curr, d_prev, d_curr, j_curr)

        buy_signal = (rsi_signal == "Uptrend") or (kdj_signal == "Bullish KDJ crossover")
        sell_signal = (rsi_signal == "Downtrend") or (kdj_signal == "Bearish KDJ crossover")

        if position is None and buy_signal:
            position = {
                'entry_time': timestamp,
                'entry_price': price,
                'entry_index': i
            }
            continue

        if position is not None:
            held_candles = i - position['entry_index']
            # Exit on sell signal or after 10 candles (~5 days)
            if sell_signal or held_candles >= 10:
                exit_price = price
                exit_time = timestamp
                pnl = exit_price - position['entry_price']
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': exit_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                position = None

    # Close any open position at last candle
    if position is not None:
        exit_price = df['close'].iloc[-1]
        exit_time = df.index[-1]
        pnl = exit_price - position['entry_price']
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl
        })

    return trades

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    all_trades = {}

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < 200:
                print(f"Not enough data for {symbol}")
                continue

            # Use last 10 candles (~5 days)
            df_recent = df.iloc[-10:].copy()

            trades = backtest_signals(df_recent)
            if trades:
                all_trades[symbol] = trades

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if all_trades:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} RSI & KDJ Backtest Results ({dt})</b>\n"]
        for coin, trades in all_trades.items():
            msg_lines.append(f"{coin} trades:")
            for t in trades:
                pnl_pct = (t['pnl'] / t['entry_price']) * 100
                msg_lines.append(
                    f"Entry: {t['entry_time'].strftime('%Y-%m-%d %H:%M')}, "
                    f"Exit: {t['exit_time'].strftime('%Y-%m-%d %H:%M')}, "
                    f"P&L: {t['pnl']:.4f} ({pnl_pct:.2f}%)"
                )
            msg_lines.append("")
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
    else:
        send_telegram_message("No trades generated in the past 5 days based on RSI & KDJ signals.")

if __name__ == "__main__":
    main()
