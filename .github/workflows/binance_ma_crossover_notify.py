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
LOOKBACK = 210       # Number of candles for indicator calculation

# For backtest: 7 days = 14 candles (12h), plus LOOKBACK for indicators
BACKTEST_CANDLES = LOOKBACK + 14

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

def calculate_kdj(df, length=5, ma1=8, ma2=8):
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-8) * 100
    k = rsv.ewm(span=ma1, adjust=False).mean()
    d = k.ewm(span=ma2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

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

# --- BACKTESTING ---

def backtest_strategy(df):
    signals = []
    returns = []

    start_idx = LOOKBACK  # Start after enough candles for indicators
    end_idx = len(df) - 1

    for i in range(start_idx, end_idx):
        sub_df = df.iloc[:i+1].copy()
        sub_df = add_indicators(sub_df)

        # Skip if MAs not ready
        if sub_df[['MA50', 'MA200', 'EMA200']].isnull().any().any():
            signals.append(0)
            returns.append(0)
            continue

        if not analyze_trend(sub_df):
            signals.append(0)
            returns.append(0)
            continue

        rsi21 = calculate_rsi(sub_df['close'], 21).iloc[-1]
        rsi13 = calculate_rsi(sub_df['close'], 13).iloc[-1]
        rsi5 = calculate_rsi(sub_df['close'], 5).iloc[-1]

        if np.isclose(rsi5, rsi13, atol=1) and np.isclose(rsi13, rsi21, atol=1):
            signals.append(0)
            returns.append(0)
            continue

        if not (rsi5 > rsi13 > rsi21 or rsi5 < rsi13 < rsi21):
            signals.append(0)
            returns.append(0)
            continue

        stoch_k, stoch_d = calculate_stochastic_rsi(sub_df, 13, 8, 5, 3)
        if np.isclose(stoch_k.iloc[-1], stoch_d.iloc[-1], atol=1):
            signals.append(0)
            returns.append(0)
            continue

        k, d, j = calculate_kdj(sub_df, 5, 8, 8)
        if np.isclose(k.iloc[-1], d.iloc[-1], atol=1) and np.isclose(d.iloc[-1], j.iloc[-1], atol=1):
            signals.append(0)
            returns.append(0)
            continue

        # Signal = Buy (1)
        signals.append(1)

        # Calculate return from current close to next close
        ret = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
        returns.append(ret)

    # Calculate cumulative return of trades (only on signal days)
    trade_returns = [r for s, r in zip(signals, returns) if s == 1]
    cumulative_return = np.prod([1 + r for r in trade_returns]) - 1 if trade_returns else 0

    return {
        'signals': signals,
        'returns': returns,
        'cumulative_return': cumulative_return
    }

# --- MAIN ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    coins_results = {}

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, BACKTEST_CANDLES)
            if len(df) < BACKTEST_CANDLES:
                print(f"Not enough data for backtest on {symbol}")
                continue

            result = backtest_strategy(df)
            coins_results[symbol] = result

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Prepare Telegram message
    if coins_results:
        msg_lines = [f"<b>KuCoin {INTERVAL.upper()} 7-Day Backtest ({dt})</b>", ""]
        for coin, res in coins_results.items():
            cum_ret_pct = res['cumulative_return'] * 100
            msg_lines.append(f"{coin}: Cumulative Return: {cum_ret_pct:.2f}%")
        send_telegram_message("\n".join(msg_lines))
    else:
        send_telegram_message("No coins had sufficient data for backtesting.")

if __name__ == "__main__":
    main()
