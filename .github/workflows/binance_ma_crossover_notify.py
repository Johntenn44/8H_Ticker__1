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
    "DOGE/USDT", "CAKE/USDT", "GRASS/USDT", "AAVE/USDT", "SUI/USDT",
    "ARB/USDT", "XLM/USDT", "MNT/USDT", "LTC/USDT", "NEAR/USDT",
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # 12-hour candles
LOOKBACK = 210       # Number of candles to fetch (>= 200)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_rsi(df, rsi_length=13, stock_length=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stock_length, min_periods=1).min()
    max_rsi = rsi.rolling(window=stock_length, min_periods=1).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-8)  # epsilon to avoid div by zero
    k = stoch_rsi.rolling(window=smooth_k, min_periods=1).mean()
    d = k.rolling(window=smooth_d, min_periods=1).mean()
    return k, d

def add_indicators(df):
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['close'].rolling(window=50, min_periods=50).mean()
    df['MA200'] = df['close'].rolling(window=200, min_periods=200).mean()
    return df

# --- TREND LOGIC ---

def analyze_trend(df):
    cp = df['close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    low = min(ma50, ema200, ma200)
    high = max(ma50, ema200, ma200)
    return low <= cp <= high

# --- DATA FETCHING ---

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
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

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    coins_meeting_all = []

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < 200:
                print(f"Not enough data for {symbol}")
                continue

            df = add_indicators(df)

            if not analyze_trend(df):
                continue

            k, d = calculate_stochastic_rsi(df, rsi_length=13, stock_length=8, smooth_k=5, smooth_d=3)
            k_last, d_last = k.iloc[-1], d.iloc[-1]
            k_prev, d_prev = k.iloc[-2], d.iloc[-2]

            # Signal: price between MAs AND Stoch RSI K crosses above D AND K is below 20 (oversold)
            if k_prev < d_prev and k_last > d_last and k_last < 20:
                coins_meeting_all.append(symbol)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if coins_meeting_all:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} Alert ({dt})</b>",
                     "Coins with price between MAs and StochRSI K crossing above D (oversold):\n"]
        msg_lines.extend(coins_meeting_all)
        send_telegram_message("\n".join(msg_lines))
    else:
        send_telegram_message("No coins satisfy conditions at this time.")

if __name__ == "__main__":
    main()
