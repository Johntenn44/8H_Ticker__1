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

# --- INDICATOR CALCULATION ---

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

# --- TREND ANALYSIS FUNCTIONS ---

def analyze_rsi_trend(rsi8, rsi13, rsi21):
    if np.isnan(rsi8) or np.isnan(rsi13) or np.isnan(rsi21):
        return "No clear RSI trend"
    if rsi8 > rsi13 > rsi21:
        return "Uptrend"
    elif rsi8 < rsi13 < rsi21:
        return "Downtrend"
    else:
        return "No clear RSI trend"

def analyze_kdj_trend(k_prev, k_curr, d_prev, d_curr, j_curr):
    if np.isnan(k_prev) or np.isnan(k_curr) or np.isnan(d_prev) or np.isnan(d_curr) or np.isnan(j_curr):
        return "No clear KDJ trend"
    if k_prev < d_prev and k_curr > d_curr and j_curr > k_curr and j_curr > d_curr:
        return "Bullish KDJ crossover"
    elif k_prev > d_prev and k_curr < d_curr and j_curr < k_curr and j_curr < d_curr:
        return "Bearish KDJ crossover"
    else:
        return "No clear KDJ trend"

def track_trend_states(signals):
    """
    Given a list of signals (e.g., 'Uptrend', 'Downtrend', 'No clear RSI trend'),
    returns a list of states: 'start', 'ongoing', 'end', or 'none' for each candle.
    """
    states = []
    prev_signal = None
    for sig in signals:
        if sig.startswith("No clear"):
            if prev_signal is not None:
                states.append("end")
                prev_signal = None
            else:
                states.append("none")
        else:
            if sig != prev_signal:
                states.append("start")
                prev_signal = sig
            else:
                states.append("ongoing")
    return states

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

    # Number of candles for 7 days (12h candles)
    period_candles = 14

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < period_candles:
                print(f"Not enough data for {symbol}")
                continue

            df_recent = df.iloc[-period_candles:].copy()

            # Calculate RSI signals per candle
            rsi8_series = calculate_rsi(df_recent['close'], 8)
            rsi13_series = calculate_rsi(df_recent['close'], 13)
            rsi21_series = calculate_rsi(df_recent['close'], 21)

            rsi_signals = []
            for i in range(len(df_recent)):
                rsi_signal = analyze_rsi_trend(rsi8_series.iloc[i], rsi13_series.iloc[i], rsi21_series.iloc[i])
                rsi_signals.append(rsi_signal)

            rsi_states = track_trend_states(rsi_signals)

            # Calculate KDJ signals per candle
            k, d, j = calculate_kdj(df_recent, length=5, ma1=8, ma2=8)
            kdj_signals = ["No clear KDJ trend"]  # first candle no prior data
            for i in range(1, len(df_recent)):
                kdj_signal = analyze_kdj_trend(k.iloc[i-1], k.iloc[i], d.iloc[i-1], d.iloc[i], j.iloc[i])
                kdj_signals.append(kdj_signal)

            kdj_states = track_trend_states(kdj_signals)

            # Latest candle trend info
            latest_rsi_signal = rsi_signals[-1]
            latest_rsi_state = rsi_states[-1]
            latest_kdj_signal = kdj_signals[-1]
            latest_kdj_state = kdj_states[-1]

            coins_with_trends[symbol] = {
                "RSI Signal": latest_rsi_signal,
                "RSI State": latest_rsi_state,
                "KDJ Signal": latest_kdj_signal,
                "KDJ State": latest_kdj_state
            }

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if coins_with_trends:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} RSI & KDJ Trend States (Last 7 Days) ({dt})</b>\n"]
        for coin, info in coins_with_trends.items():
            msg_lines.append(
                f"{coin} - RSI: {info['RSI Signal']} ({info['RSI State']}), "
                f"KDJ: {info['KDJ Signal']} ({info['KDJ State']})"
            )
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
    else:
        send_telegram_message("No coins with RSI or KDJ trends detected in the last 7 days.")

if __name__ == "__main__":
    main()
