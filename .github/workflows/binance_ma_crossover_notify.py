import os
import ccxt
import pandas as pd
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
INTERVAL = '4h'      # e.g., '4h' or '6h'
LOOKBACK = 210       # Must be >= 200
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- INDICATOR CALCULATION ---

def add_indicators(df):
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    return df

# --- TREND LOGIC ---

def analyze_trend(df):
    results = {}
    cp1 = df['close'].iloc[-1]
    cp2 = df['close'].iloc[-2]

    A1 = df['EMA8'].iloc[-1]
    B1 = df['EMA13'].iloc[-1]
    C1 = df['EMA21'].iloc[-1]
    D1 = df['EMA50'].iloc[-1]
    E1 = df['EMA200'].iloc[-1]
    MA50_1 = df['MA50'].iloc[-1]
    MA200_1 = df['MA200'].iloc[-1]

    A2 = df['EMA8'].iloc[-2]
    B2 = df['EMA13'].iloc[-2]
    C2 = df['EMA21'].iloc[-2]
    D2 = df['EMA50'].iloc[-2]
    E2 = df['EMA200'].iloc[-2]
    MA50_2 = df['MA50'].iloc[-2]
    MA200_2 = df['MA200'].iloc[-2]

    if (E1 > cp1 > A1 > B1 > C1 > D1 > MA50_1) and (cp1 < MA200_1) and \
       (E2 > cp2 > A2 > B2 > C2 > D2 > MA50_2) and (cp2 < MA200_2):
        results['start'] = 'uptrend'
    elif (E1 < cp1 < A1 < B1 < C1 < D1 < MA50_1) and (cp1 > MA200_1) and \
         (E2 < cp2 < A2 < B2 < C2 < D2 < MA50_2) and (cp2 > MA200_2):
        results['start'] = 'downtrend'

    results['values'] = {
        'cp1': cp1,
        'cp2': cp2,
        'EMA8': A1,
        'EMA13': B1,
        'EMA21': C1,
        'EMA50': D1,
        'EMA200': E1,
        'MA50': MA50_1,
        'MA200': MA200_1
    }
    return results

# --- DATA FETCHING ---

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol.replace('/', '-'), timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df

# --- CHECK TREND IN PAST N PERIODS ---

def check_trend_in_past_n_periods(df, n=10):
    """
    Check if there was any trend (uptrend/downtrend) in the past n periods.
    Returns the first found trend type ('uptrend' or 'downtrend'), or None if no trend found.
    """
    for i in range(len(df) - n, len(df)):
        if i < 2:
            continue
        window_df = df.iloc[:i+1]
        trend = analyze_trend(window_df)
        if 'start' in trend:
            return trend['start']
    return None

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bot token or chat ID not set in environment variables.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, data=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- MAIN ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    messages = []
    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < 200:
                print(f"Not enough data for {symbol}")
                continue
            df = add_indicators(df)
            trend = analyze_trend(df)
            if 'start' in trend:
                past_trend = check_trend_in_past_n_periods(df, n=10)
                vals = trend['values']
                past_trend_text = past_trend if past_trend else "None"
                msg = (
                    f"<b>Kucoin {INTERVAL.upper()} Trend Alert ({dt})</b>\n"
                    f"<b>Symbol:</b> <code>{symbol}</code>\n"
                    f"Current Trend: <b>{trend['start']}</b>\n"
                    f"Trend in past 10 periods: <b>{past_trend_text}</b>\n"
                    f"\n<code>cp1={vals['cp1']:.5f}, cp2={vals['cp2']:.5f}</code>"
                )
                messages.append(msg)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if messages:
        for msg in messages:
            send_telegram_message(msg)
    else:
        send_telegram_message("No trend signals for any coin.")

if __name__ == "__main__":
    main()
