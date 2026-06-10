import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
HOLIDAY_API_KEY = os.getenv("HOLIDAY_API_KEY")

# Location settings
CITY = os.getenv("CITY", "Chennai")
COUNTRY = os.getenv("COUNTRY", "IN")

# API Base URLs
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
HOLIDAY_BASE_URL = "https://calendarific.com/api/v2"

# Forecasting settings
FORECAST_DAYS = 30
HISTORY_DAYS = 365
TARGET_COLUMN = "sales"

# Data paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
OUTPUT_PATH = "data/outputs/"

CATEGORIES = ["Electronics", "Clothing", "Groceries", "Footwear", "HomeDecor"]