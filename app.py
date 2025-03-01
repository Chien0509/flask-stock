from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import ta

app = Flask(__name__)

# **計算技術指標**
def calculate_indicators(stock):
    try:
        data = yf.download(stock, period="6mo", interval="1d")
        if data.empty:
            return {"status": "error", "message": "無法獲取數據"}

        # 計算均線
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["MA20"] = data["Close"].rolling(window=20).mean()

        # 計算 RSI、MACD、KD
        data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()
        macd = ta.trend.MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["MACD_signal"] = macd.macd_signal()
        stoch = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"])
        data["KD_K"] = stoch.stoch()
        data["KD_D"] = stoch.stoch_signal()

        latest = data.iloc[-1]
        
        # **波段選股條件**
        rsi_value = latest["RSI"]
        macd_value = latest["MACD"]
        macd_signal_value = latest["MACD_signal"]
        kd_k_value = latest["KD_K"]
        kd_d_value = latest["KD_D"]
        ma5 = latest["MA5"]
        ma10 = latest["MA10"]
        ma20 = latest["MA20"]

        if (rsi_value > 40 and macd_value > macd_signal_value and kd_k_value > kd_d_value and ma5 > ma10 > ma20):
            signal = "BUY ✅"
        else:
            signal = "HOLD ⏳"

        return {
            "status": "success",
            "stock": stock,
            "signal": signal,
            "rsi": round(rsi_value, 2),
            "macd": round(macd_value, 2),
            "macd_signal": round(macd_signal_value, 2),
            "kd_k": round(kd_k_value, 2),
            "kd_d": round(kd_d_value, 2),
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# **首頁**
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_code = request.form.get("stock_code")
        result = calculate_indicators(stock_code)
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

