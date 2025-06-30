import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---

COINS = ["EURUSDT"]      # Binance uses no slash for symbols
EXCHANGE_ID = 'binance'  # Change to 'kraken' if you prefer Kraken
INTERVAL = '15m'         # 15-minute candles
LOOKBACK = 210           # Number of candles to fetch (>= 200)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATION ---

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

# --- TREND LOGIC ---

def analyze_trend(df):
    cp = df['close'].iloc[-1]
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

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class()
    exchange.load_markets()

    # Check if symbol exists on exchange
    if symbol not in exchange.symbols:
        raise ValueError(f"Symbol {symbol} not available on {EXCHANGE_ID}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
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
    trend_indications = {}

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < 200:
                print(f"Not enough data for {symbol}")
                continue

            df = add_indicators(df)
            trend = analyze_trend(df)
            if not trend.get('price_between_mas'):
                print(f"Price not between MAs for {symbol}, skipping.")
                continue

            rsi8 = calculate_rsi(df['close'], 8).iloc[-1]
            rsi13 = calculate_rsi(df['close'], 13).iloc[-1]
            rsi21 = calculate_rsi(df['close'], 21).iloc[-1]

            if np.isclose(rsi8, rsi13) and np.isclose(rsi13, rsi21):
                print(f"RSI values too close for {symbol}, skipping.")
                continue

            rsi_trend = analyze_rsi_trend(rsi8, rsi13, rsi21)

            k, d, j = calculate_kdj(df, length=5, ma1=8, ma2=8)

            if np.isclose(k.iloc[-1], d.iloc[-1]) and np.isclose(d.iloc[-1], j.iloc[-1]):
                print(f"KDJ values too close for {symbol}, skipping.")
                continue

            kdj_trend = analyze_kdj_trend(k, d, j)

            if rsi_trend == "No clear RSI trend" and kdj_trend == "No clear KDJ trend":
                print(f"No clear trend for {symbol}, skipping.")
                continue

            coins_meeting_all.append(symbol)
            trend_indications[symbol] = f"RSI: {rsi_trend} | KDJ: {kdj_trend}"

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if coins_meeting_all:
        msg_lines = [f"<b>{EXCHANGE_ID.capitalize()} {INTERVAL.upper()} Combined RSI & KDJ Alert ({dt})</b>",
                     "Coins satisfying conditions:\n"]
        for coin in coins_meeting_all:
            msg_lines.append(f"{coin} - {trend_indications.get(coin, 'N/A')}")
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
        print("Alert sent.")
    else:
        send_telegram_message(f"No coins satisfy the RSI and KDJ trend conditions at this time ({dt}).")
        print("No coins met conditions.")

if __name__ == "__main__":
    main()
