import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---

COINS = [
    "XRP/USDT",
    "XMR/USDT",
    "GMX/USDT",
    "LUNA/USDT",
    "TRX/USDT",
    "EIGEN/USDT",
    "APE/USDT",
    "WAVES/USDT",
    "PLUME/USDT",
    "SUSHI/USDT",
    "DOGE/USDT",
    "VIRTUAL/USDT",
    "CAKE/USDT",
    "GRASS/USDT",
    "AAVE/USDT",
    "SUI/USDT",
    "ARB/USDT",
    "XLM/USDT",
    "MNT/USDT",
    "LTC/USDT",
    "NEAR/USDT",
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # 12-hour candles
LOOKBACK = 210       # Number of candles to fetch (>= 50 for smoothing)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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

def add_ema(df):
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
    return df

def calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, k_smooth=5, d_smooth=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    stoch_rsi_raw = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi_k = stoch_rsi_raw.rolling(window=k_smooth).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()
    return stoch_rsi_k, stoch_rsi_d

# --- TREND LOGIC ---

def analyze_trend(df):
    ema8 = df['EMA8'].iloc[-1]
    ema13 = df['EMA13'].iloc[-1]
    price = df['close'].iloc[-1]

    # Example trend logic: EMA8 above EMA13 means uptrend
    ema_trend_up = ema8 > ema13

    # Price between EMA8 and EMA13 (optional filter)
    low = min(ema8, ema13)
    high = max(ema8, ema13)
    price_between_emas = low <= price <= high

    return {
        "ema_trend_up": ema_trend_up,
        "price_between_emas": price_between_emas
    }

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
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bot token or chat ID not set. Skipping Telegram message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, data=payload)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    coins_meeting_all = []
    trend_indications = {}

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < 50:
                print(f"Not enough data for {symbol}")
                continue

            df = add_ema(df)
            stoch_k, stoch_d = calculate_stoch_rsi(df)

            # Get latest values
            ema8 = df['EMA8'].iloc[-1]
            ema13 = df['EMA13'].iloc[-1]
            k_last = stoch_k.iloc[-1]
            d_last = stoch_d.iloc[-1]
            price = df['close'].iloc[-1]

            # Skip if any NaN due to rolling windows
            if np.isnan(k_last) or np.isnan(d_last) or np.isnan(ema8) or np.isnan(ema13):
                print(f"NaN values encountered for {symbol}, skipping")
                continue

            trend = analyze_trend(df)

            # Define conditions:
            # 1) EMA trend up (EMA8 > EMA13)
            # 2) Stoch RSI oversold: %K and %D below 0.2 (buy signal)
            # 3) Price between EMA8 and EMA13 (optional)
            if trend['ema_trend_up'] and k_last < 0.2 and d_last < 0.2 and trend['price_between_emas']:
                coins_meeting_all.append(symbol)
                trend_indications[symbol] = "EMA Uptrend + Stoch RSI Oversold"

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if coins_meeting_all:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} EMA + Stoch RSI Alert ({dt})</b>",
                     "Coins satisfying EMA(8>13) and Stoch RSI oversold conditions:\n"]
        for coin in coins_meeting_all:
            msg_lines.append(f"{coin} - {trend_indications.get(coin, 'N/A')}")
        msg = "\n".join(msg_lines)
        print(msg)  # Print to console
        send_telegram_message(msg)
    else:
        no_signal_msg = f"No coins satisfy EMA + Stoch RSI conditions at {dt}."
        print(no_signal_msg)
        send_telegram_message(no_signal_msg)

if __name__ == "__main__":
    main()
