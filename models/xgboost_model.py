import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_PATH

FEATURE_COLS = [
    "day_of_week", "month", "quarter", "week_of_year",
    "day_of_month", "is_weekend", "is_month_end",
    "is_month_start", "is_festival_season", "is_summer",
    "is_monsoon", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_holiday", "is_pre_holiday", "is_post_holiday",
    "days_to_next_holiday", "holiday_lift_score",
    "temperature", "humidity", "is_rainy",
    "rain_impact_score", "high_humidity", "heat_index",
    "sales_rolling_7d", "sales_rolling_30d"
]

def train_xgboost(master_df, category=None):
    """Train XGBoost model per category"""

    df = master_df.copy()

    if category:
        df = df[df["category"] == category]
        print(f"  Training XGBoost for: {category} ({len(df)} rows)")
    else:
        print(f"  Training XGBoost on all categories ({len(df)} rows)")

    # Always reset index so everything is 0-based
    df = df.dropna(subset=["sales_rolling_7d", "sales_rolling_30d"])
    df = df.reset_index(drop=True)

    # Encode category
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])
    features = FEATURE_COLS + ["category_encoded"]

    X = df[features].fillna(0)
    y = df["sales"]

    # Split using positional indices
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"    MAE  : {mae:,.0f}")
    print(f"    RMSE : {rmse:,.0f}")
    print(f"    R²   : {r2:.4f}")

    importance = pd.DataFrame({
        "feature"   : features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Save model
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    name       = category.replace(" ", "_") if category else "all"
    model_path = os.path.join(OUTPUT_PATH, f"xgb_{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le,
                     "features": features}, f)

    # Use positional indexing — no index mismatch possible
    return {
        "model"        : model,
        "label_encoder": le,
        "features"     : features,
        "metrics"      : {"MAE": mae, "RMSE": rmse, "R2": r2},
        "importance"   : importance,
        "predictions"  : pd.DataFrame({
            "date"     : df.iloc[test_idx]["date"].values,
            "actual"   : y_test.values,
            "predicted": y_pred,
            "category" : df.iloc[test_idx]["category"].values
        })
    }

def predict_xgboost(model_dict, input_df):
    model  = model_dict["model"]
    le     = model_dict["label_encoder"]
    feats  = model_dict["features"]
    df     = input_df.copy()
    df["category_encoded"] = le.transform(df["category"])
    X = df[feats].fillna(0)
    return model.predict(X)