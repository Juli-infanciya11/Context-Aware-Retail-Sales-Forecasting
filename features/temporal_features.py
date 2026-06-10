import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_temporal_features(df):
    """Create time-based features for retail patterns"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Basic time features
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_name"]     = df["date"].dt.day_name()
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["year"]         = df["date"].dt.year

    # Weekend flag (big sales driver in retail)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    # Month-end flag (salary week = higher spending)
    df["is_month_end"] = (df["date"].dt.day >= 25).astype(int)

    # Month-start flag
    df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)

    # Indian festival season flags
    df["is_festival_season"] = df["month"].isin([10, 11, 12]).astype(int)
    df["is_summer"]           = df["month"].isin([4, 5, 6]).astype(int)
    df["is_monsoon"]          = df["month"].isin([7, 8, 9]).astype(int)

    # Cyclical encoding (so model knows Dec → Jan is continuous)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Sales lag features (per category)
    df = df.sort_values(["category", "date"])
    df["sales_lag_7d"]  = df.groupby("category")["sales"].shift(7)
    df["sales_lag_14d"] = df.groupby("category")["sales"].shift(14)
    df["sales_lag_30d"] = df.groupby("category")["sales"].shift(30)

    # Rolling averages per category
    df["sales_rolling_7d"]  = df.groupby("category")["sales"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["sales_rolling_30d"] = df.groupby("category")["sales"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    print(f"  Temporal features created: {len(df.columns)} columns, {len(df)} rows")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/sales_data.csv")
    result = create_temporal_features(df)
    print(result.head())