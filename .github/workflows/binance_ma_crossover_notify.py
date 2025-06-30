import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- CONFIGURATION ---

COINS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT",
    "EIGEN/USDT", "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT",
    "DOGE/USDT", "VIRTUAL/USDT", "CAKE/USDT", "GRASS/USDT", "AAVE/USDT",
    "SUI/USDT", "ARB/USDT", "XLM/USDT", "MNT/USDT", "LTC/USDT", "NEAR/USDT",
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # 12-hour candles
LOOKBACK = 500        # Fetch enough data for indicator calculation and 10 days backtest

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

# --- BACKTEST LOGIC ---

def backtest_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3):
    k, d = calculate_stoch_rsi(df, rsi_length, stoch_length, smooth_k, smooth_d)

    position = 0  # 0 = no position, 1 = long
    entry_price = 0.0
    returns = []

    # We'll backtest only on last 10 days of data (filtered by timestamp)
    # 10 days * 2 candles per day (12h interval) = 20 candles approx
    backtest_start = df.index[-1] - timedelta(days=10)
    df_bt = df.loc[df.index >= backtest_start]

    k_bt = k.loc[df_bt.index]
    d_bt = d.loc[df_bt.index]

    for i in range(1, len(df_bt)):
        # Skip if any NaN in indicators
        if pd.isna(k_bt.iloc[i-1]) or pd.isna(d_bt.iloc[i-1]) or pd.isna(k_bt.iloc[i]) or pd.isna(d_bt.iloc[i]):
            returns.append(0)
            continue

        # Entry condition: %K crosses above %D and %K < 80
        if position == 0 and k_bt.iloc[i-1] < d_bt.iloc[i-1] and k_bt.iloc[i] > d_bt.iloc[i] and k_bt.iloc[i] < 80:
            position = 1
            entry_price = df_bt['close'].iloc[i]
            returns.append(0)  # no return on entry candle

        # Exit condition: %K crosses below %D and %K > 20
        elif position == 1 and k_bt.iloc[i-1] > d_bt.iloc[i-1] and k_bt.iloc[i] < d_bt.iloc[i] and k_bt.iloc[i] > 20:
            exit_price = df_bt['close'].iloc[i]
            ret = (exit_price - entry_price) / entry_price
            returns.append(ret)
            position = 0
            entry_price = 0.0

        else:
            returns.append(0)  # no trade or holding position without exit

    # If still holding position at end, close at last price
    if position == 1:
        exit_price = df_bt['close'].iloc[-1]
        ret = (exit_price - entry_price) / entry_price
        returns[-1] += ret  # add return to last candle

    # Calculate cumulative return
    cumulative_return = np.prod([1 + r for r in returns]) - 1

    return cumulative_return, returns, df_bt.index[1:]  # returns aligned with index from second candle

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
    results = {}

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < LOOKBACK:
                print(f"Not enough data for {symbol}")
                continue

            cum_ret, returns, timestamps = backtest_stoch_rsi(df)
            results[symbol] = cum_ret

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Prepare message with backtest results
    if results:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} Stochastic RSI 10-Day Backtest ({dt})</b>",
                     "Cumulative returns from backtest:\n"]
        for coin, ret in sorted(results.items(), key=lambda x: x[1], reverse=True):
            msg_lines.append(f"{coin}: {ret*100:.2f}%")
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
    else:
        send_telegram_message("No backtest results available for the selected coins.")

if __name__ == "__main__":
    main()
