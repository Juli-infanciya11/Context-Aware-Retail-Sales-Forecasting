import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_holiday_features(sales_df, holidays_df):
    """Add holiday context features to sales data"""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    holiday_dates = set(holidays_df["date"].dt.date)

    # Core holiday flags
    df["is_holiday"] = df["date"].dt.date.isin(holiday_dates).astype(int)

    df["is_pre_holiday"] = df["date"].apply(
        lambda d: 1 if (d + pd.Timedelta(days=1)).date() in holiday_dates else 0
    )

    df["is_post_holiday"] = df["date"].apply(
        lambda d: 1 if (d - pd.Timedelta(days=1)).date() in holiday_dates else 0
    )

    # Days until next holiday
    df["days_to_next_holiday"] = df["date"].apply(
        lambda d: min(
            [(h - d.date()).days for h in holiday_dates if h >= d.date()],
            default=30
        )
    )

    # Holiday type encoding
    holiday_type_map = dict(zip(
        holidays_df["date"].dt.date,
        holidays_df["holiday_type"]
    ))
    df["holiday_type"] = df["date"].dt.date.map(holiday_type_map).fillna("None")

    # Holiday sales lift score
    df["holiday_lift_score"] = df.apply(
        lambda r: 0.5  if r["is_holiday"] else
                  0.25 if r["is_pre_holiday"] else
                  0.15 if r["is_post_holiday"] else
                  0.3  if r["days_to_next_holiday"] <= 3 else 0.0,
        axis=1
    )

    print(f"  Holiday features created: {len(df.columns)} columns, {len(df)} rows")
    return df

if __name__ == "__main__":
    sales = pd.read_csv("data/raw/sales_data.csv")
    holidays = pd.read_csv("data/raw/holidays.csv")
    result = create_holiday_features(sales, holidays)
    print(result[["date","is_holiday","is_pre_holiday",
                  "days_to_next_holiday","holiday_lift_score"]].head(10))