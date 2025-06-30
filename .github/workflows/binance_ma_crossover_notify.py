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

# --- INDICATOR CALCULATION ---

def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100

    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

# --- TREND LOGIC ---

def analyze_stoch_rsi_trend(k, d):
    # Detect %K crossing above %D for Uptrend (with %K < 80)
    # Detect %K crossing below %D for Downtrend (with %K > 20)
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "Uptrend"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "Downtrend"
    else:
        return "No clear Stoch RSI trend"

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

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    coins_with_trends = {}

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < max(13, 8, 5, 3):
                print(f"Not enough data for {symbol}")
                continue

            k, d = calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3)
            trend = analyze_stoch_rsi_trend(k, d)

            if trend != "No clear Stoch RSI trend":
                coins_with_trends[symbol] = trend

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if coins_with_trends:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} Stochastic RSI Alert ({dt})</b>",
                     "Coins with Stochastic RSI trends:\n"]
        for coin, trend in coins_with_trends.items():
            msg_lines.append(f"{coin} - Trend: {trend}")
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
    else:
        send_telegram_message("No coins satisfy the Stochastic RSI trend conditions at this time.")

if __name__ == "__main__":
    main()
