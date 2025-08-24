import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_squared_error
from math import sqrt

def data_collection(ticker: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(
        start="2019-01-01",
        end="2025-01-01",
        auto_adjust=True
    )
    if hist is None or hist.empty:
        raise ValueError(f"No data returned by yfinance for {ticker}")

    
    y_series = pd.to_numeric(hist["Close"], errors="coerce").ffill()

    
    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)   

   
    out = pd.DataFrame({
        "ds": pd.to_datetime(idx),  
        "y": y_series
    }).dropna(subset=["y"]).reset_index(drop=True)

    return out


def main():
    # 支援命令列：python stock_prophet.py AAPL
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    df = data_collection(ticker)

    # train / test split
    training_size = int(len(df) * 0.8)
    training_data = df.iloc[:training_size].copy()
    testing_data  = df.iloc[training_size:].copy()

    # fit Prophet
    model = Prophet()
    model.fit(training_data)

    
    forecast = model.predict(testing_data[["ds"]])

    
    y_true = testing_data["y"].to_numpy()
    y_pred = forecast["yhat"].to_numpy()
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.4f}")

    
    plt.figure(figsize=(10, 5))
    plt.plot(training_data["ds"], training_data["y"], label="Training")
    plt.plot(testing_data["ds"], testing_data["y"], label="Testing")
    plt.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} - Prophet Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
