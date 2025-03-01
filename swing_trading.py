import pandas as pd
import numpy as np
import yfinance as yf
import ta

# **台股熱門股票清單**
def get_taiwan_stock_list():
    return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "3008.TW", "3034.TW", "0050.TW", "00878.TW"]

# **下載股票數據**
def get_stock_data(stock):
    try:
        data = yf.download(stock, period="6mo", interval="1d")
        if data.empty:
            print(f"⚠️ 無法獲取 {stock} 的數據")
            return None
        return data
    except Exception as e:
        print(f"❌ 取得 {stock} 數據時發生錯誤：{e}")
        return None

# **計算技術指標**
def calculate_indicators(data):
    data = data.copy()

    if "Close" in data.columns:
        close_series = data["Close"].astype(float).squeeze()
    else:
        return None

    if "High" in data.columns:
        high_series = data["High"].astype(float).squeeze()
    else:
        return None

    if "Low" in data.columns:
        low_series = data["Low"].astype(float).squeeze()
    else:
        return None

    # 計算 5 日、10 日、20 日均線
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()

    # 計算 RSI（14日相對強弱指標）
    data["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

    # 計算 MACD 指標
    macd = ta.trend.MACD(close=close_series)
    data["MACD"] = macd.macd()
    data["MACD_signal"] = macd.macd_signal()

    # 計算 KD 指標
    stoch = ta.momentum.StochasticOscillator(high=high_series, low=low_series, close=close_series)
    data["KD_K"] = stoch.stoch()
    data["KD_D"] = stoch.stoch_signal()

    return data.fillna(0)

# **波段選股策略（改進版）**
def select_swing_stocks():
    all_stocks = get_taiwan_stock_list()  # 從台股熱門股票中篩選
    selected = []

    for stock in all_stocks:
        data = get_stock_data(stock)
        if data is None or data.empty:
            continue

        data = calculate_indicators(data)
        if data is None or data.empty:
            continue

        latest = data.iloc[-1:]
        rsi_value = float(latest["RSI"].iloc[0]) if not pd.isna(latest["RSI"].iloc[0]) else 0
        macd_value = float(latest["MACD"].iloc[0]) if not pd.isna(latest["MACD"].iloc[0]) else 0
        macd_signal_value = float(latest["MACD_signal"].iloc[0]) if not pd.isna(latest["MACD_signal"].iloc[0]) else 0
        kd_k_value = float(latest["KD_K"].iloc[0]) if not pd.isna(latest["KD_K"].iloc[0]) else 0
        kd_d_value = float(latest["KD_D"].iloc[0]) if not pd.isna(latest["KD_D"].iloc[0]) else 0
        ma5 = float(latest["MA5"].iloc[0]) if not pd.isna(latest["MA5"].iloc[0]) else 0
        ma10 = float(latest["MA10"].iloc[0]) if not pd.isna(latest["MA10"].iloc[0]) else 0
        ma20 = float(latest["MA20"].iloc[0]) if not pd.isna(latest["MA20"].iloc[0]) else 0

        # **放寬波段選股條件**
        if (rsi_value > 40 and macd_value > macd_signal_value and 
            kd_k_value > kd_d_value and ma5 > ma10 and ma10 > ma20):
            selected.append({"Stock": stock, "Signal": "BUY"})

    if len(selected) == 0:
        print("⚠️ 目前市場沒有適合的波段股票（條件可能過嚴）")
        return pd.DataFrame(columns=["Stock", "Signal"])
    
    selected_df = pd.DataFrame(selected)
    return selected_df


# **主程式**
def main():
    print("🔥 AI 波段選股系統啟動！🚀")
    best_stocks = select_swing_stocks()
    if best_stocks.empty:
        print("⚠️ 沒有符合條件的波段股票")
    else:
        print(best_stocks)

if __name__ == "__main__":
    main()
