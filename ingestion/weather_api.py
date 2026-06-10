import requests
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WEATHER_API_KEY, CITY, WEATHER_BASE_URL

def fetch_current_weather():
    """Fetch current weather for the configured city"""
    try:
        url = f"{WEATHER_BASE_URL}/weather"
        params = {
            "q": CITY,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        weather = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "city": CITY,
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "weather_condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "visibility": data.get("visibility", 0) / 1000,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"  Weather fetched: {weather['temperature']}°C, {weather['description']}")
        return weather

    except Exception as e:
        print(f"  Weather API error: {e}")
        return None

def fetch_weather_forecast():
    """Fetch 5-day weather forecast"""
    try:
        url = f"{WEATHER_BASE_URL}/forecast"
        params = {
            "q": CITY,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "cnt": 40
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        records = []
        for item in data["list"]:
            records.append({
                "date": item["dt_txt"].split(" ")[0],
                "temperature": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "weather_condition": item["weather"][0]["main"],
                "wind_speed": item["wind"]["speed"],
                "rain_probability": item.get("pop", 0) * 100
            })

        df = pd.DataFrame(records)
        df = df.groupby("date").agg({
            "temperature": "mean",
            "humidity": "mean",
            "weather_condition": "first",
            "wind_speed": "mean",
            "rain_probability": "max"
        }).reset_index()

        print(f"  Forecast fetched: {len(df)} days")
        return df

    except Exception as e:
        print(f"  Weather Forecast error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Testing Weather API...")
    current = fetch_current_weather()
    forecast = fetch_weather_forecast()
    print(forecast.head())