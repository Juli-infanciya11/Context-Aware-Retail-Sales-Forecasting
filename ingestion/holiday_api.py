import requests
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HOLIDAY_API_KEY, COUNTRY

def fetch_holidays(year=None):
    """Fetch public holidays for India"""
    if year is None:
        year = datetime.now().year
    try:
        url = "https://calendarific.com/api/v2/holidays"
        params = {
            "api_key": HOLIDAY_API_KEY,
            "country": COUNTRY,
            "year": year
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        holidays = []
        for h in data["response"]["holidays"]:
            holidays.append({
                "date": h["date"]["iso"][:10],
                "holiday_name": h["name"],
                "holiday_type": h["type"][0] if h["type"] else "Other",
                "description": h.get("description", "")[:100]
            })

        df = pd.DataFrame(holidays)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        print(f"  Holidays fetched: {len(df)} holidays for {year}")
        return df

    except Exception as e:
        print(f"  Holiday API error: {e}")
        return pd.DataFrame()

def get_holiday_context(target_date, holidays_df):
    """Check if a date is a holiday, pre-holiday, or post-holiday"""
    if holidays_df.empty:
        return {"is_holiday": 0, "is_pre_holiday": 0, "is_post_holiday": 0, "holiday_name": None}

    target = pd.to_datetime(target_date)
    holiday_dates = pd.to_datetime(holidays_df["date"])

    is_holiday = int(target in holiday_dates.values)

    from pandas.tseries.offsets import Day
    is_pre_holiday = int((target + Day(1)) in holiday_dates.values)
    is_post_holiday = int((target - Day(1)) in holiday_dates.values)

    holiday_name = None
    if is_holiday:
        holiday_name = holidays_df[holidays_df["date"] == target]["holiday_name"].values[0]

    return {
        "is_holiday": is_holiday,
        "is_pre_holiday": is_pre_holiday,
        "is_post_holiday": is_post_holiday,
        "holiday_name": holiday_name
    }

if __name__ == "__main__":
    print("Testing Holiday API...")
    df = fetch_holidays()
    print(df.head(10))