import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import math

# --- CONFIGURATION ---

SYMBOL = "EURUSD=X"  # Yahoo Finance ticker for EUR/USD forex pair
INTERVAL = '15m'     # 15-minute candles
LOOKBACK = 210       # Number of candles to fetch (about 52.5 hours)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATION ---

def add_indicators(df):
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
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
    low_min = df['Low'].rolling(window=length, min_periods=1).min()
    high_max = df['High'].rolling(window=length, min_periods=1).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# --- TREND LOGIC ---

def analyze_trend(df):
    cp = df['Close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]

    low = min(ma50, ema200, ma200)
    high = max(ma50, ema200, ma200)

    results = {}
    results['price_between_mas'] = low <= cp <= high
    return results

def analyze_rsi_trend(rsi8, rsi13, rsi21):
    if rsi8 > rsi13 > rsi21:
        return "Uptrend"
    elif rsi8 < rsi13 < rsi21:
        return "Downtrend"
    else:
        return "No clear RSI trend"

def analyze_kdj_trend(k, d, j):
    if len(k) < 2 or len(d) < 2 or len(j) < 2:
        return "No clear KDJ trend"
    k_prev, k_curr = k.iloc[-2], k.iloc[-1]
    d_prev, d_curr = d.iloc[-2], d.iloc[-1]
    j_prev, j_curr = j.iloc[-2], j.iloc[-1]

    if k_prev < d_prev and k_curr > d_curr and j_curr > k_curr and j_curr > d_curr:
        return "Bullish KDJ crossover"
    elif k_prev > d_prev and k_curr < d_curr and j_curr < k_curr and j_curr < d_curr:
        return "Bearish KDJ crossover"
    else:
        return "No clear KDJ trend"

# --- DATA FETCHING ---

def fetch_ohlcv_yfinance(symbol, interval, lookback):
    # Calculate number of days to cover lookback candles at 15m interval
    minutes_per_candle = 15
    total_minutes = lookback * minutes_per_candle
    days = max(1, math.ceil(total_minutes / (60 * 24)))  # at least 1 day

    period_str = f"{days}d"

    df = yf.download(
        symbol,
        period=period_str,
        interval=interval,
        progress=False,
        auto_adjust=False  # set True if you want adjusted prices
    )

    if df.empty:
        raise ValueError(f"No data fetched for {symbol} with interval {interval}")

    # Keep only last LOOKBACK candles
    df = df.tail(lookback)

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
    try:
        df = fetch_ohlcv_yfinance(SYMBOL, INTERVAL, LOOKBACK)
        if len(df) < 200:
            print(f"Not enough data for {SYMBOL}")
            return

        df = add_indicators(df)
        trend = analyze_trend(df)
        if not trend.get('price_between_mas'):
            print("Price not between MAs, skipping alert.")
            return

        rsi8 = calculate_rsi(df['Close'], 8).iloc[-1]
        rsi13 = calculate_rsi(df['Close'], 13).iloc[-1]
        rsi21 = calculate_rsi(df['Close'], 21).iloc[-1]

        if np.isclose(rsi8, rsi13) and np.isclose(rsi13, rsi21):
            print("RSI values too close, skipping alert.")
            return

        rsi_trend = analyze_rsi_trend(rsi8, rsi13, rsi21)

        k, d, j = calculate_kdj(df, length=5, ma1=8, ma2=8)

        if np.isclose(k.iloc[-1], d.iloc[-1]) and np.isclose(d.iloc[-1], j.iloc[-1]):
            print("KDJ values too close, skipping alert.")
            return

        kdj_trend = analyze_kdj_trend(k, d, j)

        if rsi_trend == "No clear RSI trend" and kdj_trend == "No clear KDJ trend":
            print("No clear trend detected, skipping alert.")
            return

        msg = (f"<b>Yahoo Finance {INTERVAL.upper()} Combined RSI & KDJ Alert ({dt})</b>\n"
               f"{SYMBOL} - RSI: {rsi_trend} | KDJ: {kdj_trend}")

        send_telegram_message(msg)
        print("Alert sent:", msg)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
