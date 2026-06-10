# print("=== Retail Sales Forecasting Pipeline ===")
# print("Step 1: Project setup complete")
# print(f"Config loaded. City: Chennai | Country: IN")
# print("Ready to connect APIs in Step 2...")

import os
from datetime import datetime

print("=" * 50)
print("  Retail Sales Forecasting Pipeline")
print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

print("\n[Step 2] Fetching live data from APIs...")

# Weather
print("\n  Fetching weather data...")
from ingestion.weather_api import fetch_current_weather, fetch_weather_forecast
current_weather = fetch_current_weather()
weather_forecast = fetch_weather_forecast()

# Holidays
print("\n  Fetching holiday data...")
from ingestion.holiday_api import fetch_holidays
holidays = fetch_holidays()

# Sales
print("\n  Generating sales data...")
from ingestion.sales_api import generate_sales_data
sales = generate_sales_data()

# Save to raw folder
print("\n  Saving raw data...")
weather_forecast.to_csv("data/raw/weather_forecast.csv", index=False)
holidays.to_csv("data/raw/holidays.csv", index=False)
sales.to_csv("data/raw/sales_data.csv", index=False)

print("\n  Files saved to data/raw/")
print("\n[Step 2 Complete] All APIs working!")
print(f"  Weather: {current_weather['temperature']}°C in Chennai")
print(f"  Holidays: {len(holidays)} holidays loaded")
print(f"  Sales records: {len(sales)} rows generated")


print("\n[Step 3] Building master feature dataset...")
from features.merge_features import build_master_dataset
master = build_master_dataset()

print("\n" + "=" * 50)
print("  PIPELINE SUMMARY")
print("=" * 50)
print(f"  Raw weather rows   : {len(weather_forecast)}")
print(f"  Raw holiday rows   : {len(holidays)}")
print(f"  Raw sales rows     : {len(sales)}")
print(f"  Master dataset     : {master.shape[0]} rows x {master.shape[1]} columns")
print("=" * 50)


# ── STEP 4 ── (append below Step 3)
print("\n[Step 4] Training ML Models...")
from models.ensemble import train_all_models
model_results, model_summary = train_all_models(master)

print("\n[Step 4 Complete] All models trained!")
print("\n[Step 4 Complete] All models trained!")
print(f"  Models saved to: data/outputs/")

# ── STEP 5 ── (append below Step 4)
print("\n[Step 5] Starting FastAPI server...")
print("  API will be available at: http://localhost:8000")
print("  Interactive docs at    : http://localhost:8000/docs")
print("  Press Ctrl+C to stop the server\n")

import uvicorn
uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)