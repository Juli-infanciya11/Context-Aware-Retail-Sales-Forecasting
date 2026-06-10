import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HISTORY_DAYS, RAW_DATA_PATH

# Retail categories for Chennai market
CATEGORIES = ["Electronics", "Clothing", "Groceries", "Footwear", "HomeDecor"]

def generate_sales_data(days=None):
    """Generate realistic mock retail sales — changes every load"""
    if days is None:
        days = HISTORY_DAYS

    random.seed(42)  # ensures different data every run
    np.random.seed(42)

    records = []
    start_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        current_date = start_date + timedelta(days=i)
        day_of_week = current_date.weekday()
        month = current_date.month

        for category in CATEGORIES:
            # Base sales per category
            base = {"Electronics": 45000, "Clothing": 30000,
                    "Groceries": 25000, "Footwear": 20000, "HomeDecor": 15000}[category]

            # Weekend boost
            weekend_factor = 1.35 if day_of_week >= 5 else 1.0

            # Seasonal boost (festival season Oct-Dec in India)
            seasonal_factor = 1.5 if month in [10, 11, 12] else \
                              1.2 if month in [8, 9] else \
                              0.85 if month in [4, 5, 6] else 1.0

            # Random daily noise
            noise = np.random.normal(1.0, 0.1)

            sales = base * weekend_factor * seasonal_factor * noise
            sales = max(0, round(sales, 2))

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "category": category,
                "sales": sales,
                "transactions": int(sales / random.randint(800, 1200)),
                "avg_basket_size": round(random.uniform(600, 2500), 2),
                "day_of_week": day_of_week,
                "month": month
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Sales data generated: {len(df)} records ({days} days × {len(CATEGORIES)} categories)")
    return df

if __name__ == "__main__":
    print("Testing Sales Data Generator...")
    df = generate_sales_data()
    print(df.head(10))
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")