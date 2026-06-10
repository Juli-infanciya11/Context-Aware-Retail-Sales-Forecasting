import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_weather_features(weather_df):
    """Create ML-ready weather features from raw weather data"""
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Temperature categories
    df["temp_category"] = pd.cut(
        df["temperature"],
        bins=[0, 20, 28, 35, 50],
        labels=["cold", "pleasant", "warm", "hot"]
    )

    # Heat index (feels hotter when humid)
    df["heat_index"] = df["temperature"] + (df["humidity"] / 100) * 5

    # Rain flag
    df["is_rainy"] = df["weather_condition"].str.lower().isin(
        ["rain", "drizzle", "thunderstorm"]
    ).astype(int)

    # Rain impact on shopping (rain reduces footfall)
    df["rain_impact_score"] = df.apply(
        lambda r: -0.3 if r["is_rainy"] else
                  -0.1 if r["weather_condition"] == "Clouds" else 0.1,
        axis=1
    )

    # High humidity flag (above 80% discourages shopping)
    df["high_humidity"] = (df["humidity"] > 80).astype(int)

    # Temperature lag features (rolling averages)
    df = df.sort_values("date")
    df["temp_rolling_3d"] = df["temperature"].rolling(3, min_periods=1).mean()
    df["humidity_rolling_3d"] = df["humidity"].rolling(3, min_periods=1).mean()

    print(f"  Weather features created: {len(df.columns)} columns, {len(df)} rows")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/weather_forecast.csv")
    result = create_weather_features(df)
    print(result)