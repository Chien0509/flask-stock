import pandas as pd
import numpy as np
import yfinance as yf
import ta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# **獲取台股所有上市股票代碼**
def get_taiwan_stock_list():
    return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "3008.TW", "3034.TW", "0050.TW", "00878.TW"]

# **下載股票數據**
def get_stock_data(stock):
    try:
        data = yf.download(stock, period="12mo", interval="1d")
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

    # **計算 5 日均量**
    data["Volume_MA5"] = data["Volume"].rolling(5).mean()

    # **計算波動率**
    data["Volatility"] = data["Close"].pct_change().rolling(5).std()

    return data.fillna(0)

# **建立 AI 模型（LSTM 預測當沖機會）**
def train_lstm_model(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[["RSI", "MACD", "MACD_signal", "KD_K", "KD_D", "Volatility"]])

    X, y = [], []
    for i in range(10, len(data_scaled) - 1):
        X.append(data_scaled[i - 10:i])
        y.append(1 if data["Close"].iloc[i + 1] > data["Close"].iloc[i] else 0)  # 上漲 = 1，下降 = 0

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 6)),
        LSTM(50),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    return model, scaler

# **AI 當沖選股策略**
def select_best_stocks():
    all_stocks = get_taiwan_stock_list()
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
        volume_ma5 = float(latest["Volume_MA5"].iloc[0]) if not pd.isna(latest["Volume_MA5"].iloc[0]) else 0
        volatility = float(latest["Volatility"].iloc[0]) if not pd.isna(latest["Volatility"].iloc[0]) else 0

        # **確保成交量大於過去 5 日均量，並且波動率大於 0.02**
        if latest["Volume"].iloc[0] < volume_ma5 or volatility < 0.02:
            continue

        # **計算 AI 分數**
        score = 0
        if rsi_value > 50: score += 1
        if macd_value > macd_signal_value: score += 1
        if kd_k_value > kd_d_value: score += 1

        if score >= 2:
            selected.append({"Stock": stock, "Score": score, "Signal": "BUY" if score == 3 else "HOLD"})

    if len(selected) == 0:
        print("⚠️ 沒有符合條件的當沖股票")
        return pd.DataFrame(columns=["Stock", "Score", "Signal"])
    
    selected_df = pd.DataFrame(selected).sort_values(by="Score", ascending=False).head(5)
    return selected_df

# **主程式**
def main():
    print("🔥 AI 當沖選股系統啟動！🚀")
    best_stocks = select_best_stocks()
    if best_stocks.empty:
        print("⚠️ 沒有符合條件的當沖股票")
    else:
        print(best_stocks)

if __name__ == "__main__":
    main()
