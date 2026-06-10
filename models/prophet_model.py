import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_PATH, FORECAST_DAYS

def train_prophet(master_df, category):
    """Train SARIMA model for a specific category (replaces Prophet)"""
    print(f"  Training SARIMA for: {category}")

    df = master_df[master_df["category"] == category].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Use total daily sales as time series
    ts = df.set_index("date")["sales"]

    # Train/test split
    split      = int(len(ts) * 0.8)
    train_ts   = ts.iloc[:split]
    test_ts    = ts.iloc[split:]

    # SARIMA(1,1,1) — simple and effective for retail
    model = SARIMAX(
        train_ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)

    # Predict on test
    predictions = fitted.forecast(steps=len(test_ts))
    y_pred      = np.array(predictions)
    y_actual    = test_ts.values

    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2   = r2_score(y_actual, y_pred)

    print(f"    MAE  : {mae:,.0f}")
    print(f"    RMSE : {rmse:,.0f}")
    print(f"    R²   : {r2:.4f}")

    # Future forecast
    future_pred = fitted.forecast(steps=FORECAST_DAYS)
    future_dates = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )
    forecast_df = pd.DataFrame({
        "ds"  : future_dates,
        "yhat": future_pred.values
    })

    # Save model
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    name       = category.replace(" ", "_")
    model_path = os.path.join(OUTPUT_PATH, f"sarima_{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(fitted, f)

    return {
        "model"      : fitted,
        "metrics"    : {"MAE": mae, "RMSE": rmse, "R2": r2},
        "forecast"   : forecast_df,
        "predictions": pd.DataFrame({
            "date"     : test_ts.index,
            "actual"   : y_actual,
            "predicted": y_pred,
            "category" : category
        })
    }
    