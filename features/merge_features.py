import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.weather_features  import create_weather_features
from features.holiday_features  import create_holiday_features
from features.temporal_features import create_temporal_features
from config import PROCESSED_DATA_PATH

def build_master_dataset():
    """Merge all features into one master ML-ready dataset"""
    print("\n  Loading raw data...")
    sales    = pd.read_csv("data/raw/sales_data.csv")
    holidays = pd.read_csv("data/raw/holidays.csv")
    weather  = pd.read_csv("data/raw/weather_forecast.csv")

    print("\n  Engineering features...")
    # Step 1 — temporal features on sales
    sales_temp = create_temporal_features(sales)

    # Step 2 — holiday features
    sales_hol = create_holiday_features(sales_temp, holidays)

    # Step 3 — weather features
    weather_feat = create_weather_features(weather)

    # Step 4 — merge weather into sales on date
    weather_cols = ["date", "temperature", "humidity", "is_rainy",
                    "rain_impact_score", "high_humidity",
                    "heat_index", "temp_category",
                    "temp_rolling_3d", "humidity_rolling_3d"]

    weather_feat["date"] = pd.to_datetime(weather_feat["date"])
    sales_hol["date"]    = pd.to_datetime(sales_hol["date"])

    master = pd.merge(
        sales_hol,
        weather_feat[weather_cols],
        on="date",
        how="left"
    )

    # Fill missing weather with median (for historical dates)
    for col in ["temperature", "humidity", "heat_index"]:
        master[col] = master[col].fillna(master[col].median())
    master["is_rainy"]          = master["is_rainy"].fillna(0)
    master["rain_impact_score"] = master["rain_impact_score"].fillna(0)
    master["high_humidity"]     = master["high_humidity"].fillna(0)
    master["temp_category"]     = master["temp_category"].fillna("warm")

    # Save master dataset
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, "master_dataset.csv")
    master.to_csv(output_path, index=False)

    print(f"\n  Master dataset saved: {output_path}")
    print(f"  Shape: {master.shape[0]} rows × {master.shape[1]} columns")
    print(f"\n  Columns: {list(master.columns)}")
    return master

if __name__ == "__main__":
    df = build_master_dataset()
    print("\n  Sample:")
    print(df[["date","category","sales","temperature",
              "is_holiday","is_weekend","holiday_lift_score"]].head(10).to_string())