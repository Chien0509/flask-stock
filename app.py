from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import ta

app = Flask(__name__)

# **計算技術指標**
import yfinance as yf
import ta
import pandas as pd

def calculate_indicators(stock):
    try:
        print(f"正在獲取股票數據: {stock}")
        stock_data = yf.Ticker(stock)

        # **獲取股票名稱**
        stock_name = stock_data.info.get("longName", "未知名稱")

        # **下載過去 6 個月的日線數據**
        data = stock_data.history(period="6mo", interval="1d")

        if data.empty:
            return {"status": "error", "message": "無法獲取數據，請確認股票代號是否正確"}

        # 確保 'Close'、'High'、'Low' 欄位為 1 維 Series
        close_series = data["Close"].astype(float).squeeze()
        high_series = data["High"].astype(float).squeeze()
        low_series = data["Low"].astype(float).squeeze()

        # 計算均線（新增 30 日均線）
        data["MA5"] = close_series.rolling(window=5).mean()
        data["MA10"] = close_series.rolling(window=10).mean()
        data["MA20"] = close_series.rolling(window=20).mean()
        data["MA30"] = close_series.rolling(window=30).mean()

        # 計算 RSI
        data["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

        # 計算 MACD
        macd = ta.trend.MACD(close=close_series)
        data["MACD"] = macd.macd()
        data["MACD_signal"] = macd.macd_signal()

        # 計算 KD 指標
        stoch = ta.momentum.StochasticOscillator(high=high_series, low=low_series, close=close_series)
        data["KD_K"] = stoch.stoch()
        data["KD_D"] = stoch.stoch_signal()

        # 取得最新一筆數據
        latest = data.iloc[-1]

        # **確保變數為 `float`**
        def get_value(series):
            return series.iloc[-1] if isinstance(series, pd.Series) else series

        close_price = get_value(latest["Close"])
        ma5 = get_value(latest["MA5"])
        ma10 = get_value(latest["MA10"])
        ma20 = get_value(latest["MA20"])
        ma30 = get_value(latest["MA30"])
        rsi_value = get_value(latest["RSI"])
        macd_value = get_value(latest["MACD"])
        macd_signal_value = get_value(latest["MACD_signal"])
        kd_k_value = get_value(latest["KD_K"])
        kd_d_value = get_value(latest["KD_D"])

        # **決定交易訊號**
        if float(rsi_value) > 40 and float(macd_value) > float(macd_signal_value) and float(kd_k_value) > float(kd_d_value) and float(ma5) > float(ma10) > float(ma20) > float(ma30):
            signal = "BUY ✅"
        else:
            signal = "HOLD ⏳"

        # **改進建議買入價 / 賣出價計算**
        support = min(low_series[-20:])  # 近 20 天最低點作為支撐位
        resistance = max(high_series[-20:])  # 近 20 天最高點作為阻力位
        avg_price = (ma5 + ma10 + ma20 + ma30) / 4  # 計算短期均線的平均價格

        # **考慮當前價格，計算買入 / 賣出價**
        suggested_buy = round((support + avg_price + close_price) / 3, 2)
        suggested_sell = round((resistance + avg_price + close_price) / 3, 2)

        # **計算停利 / 停損價位**
        stop_loss = round(suggested_buy * 0.95, 2)  # 停損價 = 買入價 - 5%
        take_profit = round(suggested_buy * 1.1, 2)  # 停利價 = 買入價 + 10%

        result = {
            "status": "success",
            "stock": stock,
            "stock_name": stock_name,
            "signal": signal,
            "current_price": round(close_price, 2),
            "suggested_buy": suggested_buy,
            "suggested_sell": suggested_sell,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rsi": round(rsi_value, 2),
            "macd": round(macd_value, 2),
            "macd_signal": round(macd_signal_value, 2),
            "kd_k": round(kd_k_value, 2),
            "kd_d": round(kd_d_value, 2),
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2),
            "ma30": round(ma30, 2)
        }
        print(f"回傳結果: {result}")
        return result

    except Exception as e:
        print(f"錯誤: {str(e)}")
        return {"status": "error", "message": str(e)}

# **首頁**
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        stock_code = request.form.get("stock_code")
        print(f"收到請求，股票代號: {stock_code}")  # 確保 Flask 收到請求
        if stock_code:
            result = calculate_indicators(stock_code)
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
