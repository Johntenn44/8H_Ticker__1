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
    # Add more symbols here
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '6h'      # Candle timeframe
LOOKBACK = 210       # Number of candles to fetch (>= 200)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- INDICATOR PARAMETERS ---
KDJ_LENGTH = 5
KDJ_MA1 = 8
KDJ_MA2 = 8

RSI_PERIODS = [5, 13, 21]
WR_PERIODS = [8, 13, 50, 200]

# --- INDICATOR CALCULATION ---

def add_indicators(df):
    # EMAs and MAs
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # KDJ indicator
    df = add_kdj(df, length=KDJ_LENGTH, ma1=KDJ_MA1, ma2=KDJ_MA2)

    # RSI indicators
    for period in RSI_PERIODS:
        df[f'RSI{period}'] = rsi(df['close'], period)

    # Williams %R indicators
    for period in WR_PERIODS:
        df[f'WR{period}'] = williams_r(df['high'], df['low'], df['close'], period)

    return df

def add_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min.replace(0, 0.0001)) * 100  # avoid div0

    k = rsv.ewm(alpha=1/ma1, adjust=False).mean()
    d = k.ewm(alpha=1/ma2, adjust=False).mean()
    j = 3 * k - 2 * d

    df['K'] = k
    df['D'] = d
    df['J'] = j
    return df

def rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 0.0001)  # avoid div0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def williams_r(high, low, close, period):
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 0.0001)
    return wr

# --- TREND LOGIC ---

def analyze_trend(df):
    results = {}

    # Prerequisite condition:
    # price, EMA8, EMA13, EMA21, EMA50 all NOT above MA50, MA200, or EMA200
    cp = df['close'].iloc[-1]
    ema8 = df['EMA8'].iloc[-1]
    ema13 = df['EMA13'].iloc[-1]
    ema21 = df['EMA21'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]

    prereq = all([
        cp <= ma50, cp <= ma200, cp <= ema200,
        ema8 <= ma50, ema8 <= ma200, ema8 <= ema200,
        ema13 <= ma50, ema13 <= ma200, ema13 <= ema200,
        ema21 <= ma50, ema21 <= ma200, ema21 <= ema200,
        ema50 <= ma50, ema50 <= ma200, ema50 <= ema200
    ])

    if not prereq:
        return results  # Prerequisite not met, no trend identified

    # KDJ values
    K = df['K'].iloc[-1]
    D = df['D'].iloc[-1]
    J = df['J'].iloc[-1]

    # RSI values
    RSI5 = df['RSI5'].iloc[-1]
    RSI13 = df['RSI13'].iloc[-1]
    RSI21 = df['RSI21'].iloc[-1]

    # Williams %R values
    WR8 = df['WR8'].iloc[-1]
    WR13 = df['WR13'].iloc[-1]
    WR50 = df['WR50'].iloc[-1]
    WR200 = df['WR200'].iloc[-1]

    # Check KDJ trend condition
    kdj_up = (J > D > K)
    kdj_down = (K > D > J)

    # Check RSI trend condition
    rsi_up = (RSI5 > RSI13 > RSI21)
    rsi_down = (RSI21 > RSI13 > RSI5)

    # Check WR trend condition
    wr_up = (WR8 > WR13 > WR50 > WR200)
    wr_down = (WR200 > WR50 > WR13 > WR8)

    # Check if trend ended: 8 and 13 WR between 50 and 200 (neutral zone)
    wr_end = (50 <= WR8 <= 200) and (50 <= WR13 <= 200)

    # Identify trend start
    if kdj_up and rsi_up and wr_up and not wr_end:
        results['start'] = 'uptrend'
    elif kdj_down and rsi_down and wr_down and not wr_end:
        results['start'] = 'downtrend'

    # Identify trend end
    if wr_end:
        results['end'] = True

    # Include indicator values for debugging/alerts
    results['values'] = {
        'close': cp,
        'EMA8': ema8,
        'EMA13': ema13,
        'EMA21': ema21,
        'EMA50': ema50,
        'MA50': ma50,
        'MA200': ma200,
        'EMA200': ema200,
        'K': K,
        'D': D,
        'J': J,
        'RSI5': RSI5,
        'RSI13': RSI13,
        'RSI21': RSI21,
        'WR8': WR8,
        'WR13': WR13,
        'WR50': WR50,
        'WR200': WR200,
    }

    return results

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
                vals = trend['values']
                msg = (
                    f"<b>Kucoin {INTERVAL.upper()} Trend Alert ({dt})</b>\n"
                    f"<b>Symbol:</b> <code>{symbol}</code>\n"
                    f"Start: <b>{trend['start']}</b>\n"
                    f"\n<code>Close={vals['close']:.5f}, EMA8={vals['EMA8']:.5f}, EMA13={vals['EMA13']:.5f}, EMA21={vals['EMA21']:.5f}, EMA50={vals['EMA50']:.5f}, "
                    f"MA50={vals['MA50']:.5f}, MA200={vals['MA200']:.5f}, EMA200={vals['EMA200']:.5f}\n"
                    f"K={vals['K']:.2f}, D={vals['D']:.2f}, J={vals['J']:.2f}\n"
                    f"RSI5={vals['RSI5']:.2f}, RSI13={vals['RSI13']:.2f}, RSI21={vals['RSI21']:.2f}\n"
                    f"WR8={vals['WR8']:.2f}, WR13={vals['WR13']:.2f}, WR50={vals['WR50']:.2f}, WR200={vals['WR200']:.2f}</code>"
                )
                messages.append(msg)
            elif 'end' in trend:
                messages.append(f"<b>Kucoin {INTERVAL.upper()} Trend End Alert ({dt})</b>\n<b>Symbol:</b> <code>{symbol}</code>\nTrend ended based on WR indicator.")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if messages:
        for msg in messages:
            send_telegram_message(msg)
    else:
        send_telegram_message("No trend signals for any coin.")

if __name__ == "__main__":
    main()
