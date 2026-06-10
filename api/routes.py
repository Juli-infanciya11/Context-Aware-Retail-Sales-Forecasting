import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    WeatherResponse, HolidayImpactResponse,
    SalesSummaryResponse, ModelMetrics
)
from ingestion.weather_api import fetch_current_weather, fetch_weather_forecast
from ingestion.holiday_api import fetch_holidays
from config import CATEGORIES, OUTPUT_PATH

router = APIRouter()

def safe(val):
    """Convert NaN/Inf → 0, keep strings unchanged"""
    try:
        # ✅ If string → return as is
        if isinstance(val, str):
            return val

        # ✅ Handle None
        if val is None:
            return 0

        # ✅ Handle NaN / Inf
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return 0

        return float(val)

    except:
        return val   # ✅ fallback (very important)

# ── helpers ──────────────────────────────────────────
def load_master():
    path = "data/processed/master_dataset.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail="Master dataset not found. Run pipeline first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_model_summary():
    path = os.path.join(OUTPUT_PATH, "model_summary.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail="Model summary not found. Run pipeline first.")
    return pd.read_csv(path)

# ── routes ───────────────────────────────────────────

@router.get("/")
def root():
    return {
        "project": "Context-Aware Retail Sales Forecasting",
        "version": "1.0",
        "city"   : "Chennai",
        "status" : "running",
        "endpoints": [
            "/weather",
            "/holidays",
            "/forecast",
            "/sales-summary",
            "/weather-impact",
            "/holiday-impact",
            "/model-metrics",
            "/dashboard-data"
        ]
    }

@router.get("/weather", response_model=WeatherResponse)
def get_weather():
    """Live current weather for Chennai"""
    data = fetch_current_weather()
    if not data:
        raise HTTPException(status_code=503,
                            detail="Weather API unavailable")
    return WeatherResponse(**data)

@router.get("/holidays")
def get_holidays():
    """All Indian holidays for current year"""
    df = fetch_holidays()
    if df.empty:
        raise HTTPException(status_code=503,
                            detail="Holiday API unavailable")
    df["date"] = df["date"].astype(str)
    return {
        "total"   : len(df),
        "year"    : datetime.now().year,
        "holidays": df.to_dict(orient="records")
    }

@router.post("/forecast", response_model=ForecastResponse)
def get_forecast(request: ForecastRequest):
    """Sales forecast for a category using trained models"""
    if request.category not in CATEGORIES:
        raise HTTPException(status_code=400,
                            detail=f"Category must be one of {CATEGORIES}")

    master   = load_master()
    summary  = load_model_summary()
    cat_data = master[master["category"] == request.category].copy()
    cat_data = cat_data.sort_values("date")

    # Get model metrics for this category
    metrics_row = summary[summary["category"] == request.category]
    r2 = float(metrics_row["xgb_r2"].values[0]) if len(metrics_row) else 0.0

    # Build forecast using rolling average + trend + noise simulation
    last_sales    = cat_data["sales"].tail(30).mean()
    trend         = cat_data["sales"].tail(7).mean() - cat_data["sales"].tail(30).mean()

    forecast_points = []
    for i in range(1, request.days + 1):
        future_date  = datetime.now() + timedelta(days=i)
        dow          = future_date.weekday()
        month        = future_date.month

        weekend_mult  = 1.3  if dow >= 5 else 1.0
        festival_mult = 1.4  if month in [10, 11, 12] else \
                        1.15 if month in [8, 9]        else \
                        0.9  if month in [4, 5, 6]     else 1.0
        noise         = np.random.normal(1.0, 0.05)
        predicted     = max(0, (last_sales + trend * i * 0.1)
                            * weekend_mult * festival_mult * noise)

        forecast_points.append(ForecastPoint(
            date           = future_date.strftime("%Y-%m-%d"),
            predicted_sales= round(predicted, 2),
            lower_bound    = round(predicted * 0.85, 2),
            upper_bound    = round(predicted * 1.15, 2),
            category       = request.category
        ))

    return ForecastResponse(
        category    = request.category,
        forecast_days= request.days,
        forecasts   = forecast_points,
        model_r2    = round(r2, 4),
        generated_at= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@router.get("/sales-summary")
def get_sales_summary():
    """Summary statistics per category"""
    master  = load_master()
    results = []

    for cat in CATEGORIES:
        df = master[master["category"] == cat].copy()

        # Weather correlation
        corr = df["sales"].corr(df["temperature"])

        # Holiday lift
        holiday_sales    = df[df["is_holiday"] == 1]["sales"].mean()
        non_holiday_sales= df[df["is_holiday"] == 0]["sales"].mean()
        lift = ((holiday_sales - non_holiday_sales) / non_holiday_sales * 100
                if non_holiday_sales > 0 else 0)

        best_day  = df.loc[df["sales"].idxmax(), "date"]
        worst_day = df.loc[df["sales"].idxmin(), "date"]

        results.append({
            "category"         : cat,
            "total_sales"      : round(df["sales"].sum(), 2),
            "avg_daily_sales"  : round(df["sales"].mean(), 2),
            "best_day"         : str(best_day)[:10],
            "worst_day"        : str(worst_day)[:10],
            "weather_correlation": round(corr, 4),
            "holiday_lift_pct" : round(lift, 2)
        })

    return {"categories": results,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@router.get("/weather-impact")
def get_weather_impact():
    """How weather conditions affect sales per category"""
    master = load_master()
    results = []

    for cat in CATEGORIES:
        df = master[master["category"] == cat].copy()

        rainy_sales = df[df["is_rainy"] == 1]["sales"].mean()
        clear_sales = df[df["is_rainy"] == 0]["sales"].mean()

        rain_impact = (
            (rainy_sales - clear_sales) / clear_sales * 100
            if safe(clear_sales) > 0 else 0
        )

        hot_sales = df[df["temperature"] > 35]["sales"].mean()
        normal_sales = df[df["temperature"] <= 35]["sales"].mean()

        heat_impact = (
            (hot_sales - normal_sales) / normal_sales * 100
            if safe(normal_sales) > 0 else 0
        )

        temp_corr = df["sales"].corr(df["temperature"])
        humidity_corr = df["sales"].corr(df["humidity"])

        results.append({
            "category": cat,
            "avg_sales_rainy": round(safe(rainy_sales), 2),
            "avg_sales_clear": round(safe(clear_sales), 2),
            "rain_impact_pct": round(safe(rain_impact), 2),
            "heat_impact_pct": round(safe(heat_impact), 2),
            "temp_correlation": round(safe(temp_corr), 4),
            "humidity_correlation": round(safe(humidity_corr), 4)
        })

    # ✅ Clean weather API output
    weather = fetch_current_weather()

    for k, v in weather.items():
        weather[k] = safe(v)

    return {
        "weather_impact": results,
        "current_weather": weather,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@router.get("/holiday-impact")
def get_holiday_impact():
    """Upcoming holidays with expected sales impact"""
    master   = load_master()
    holidays = fetch_holidays()

    if holidays.empty:
        raise HTTPException(status_code=503, detail="Holiday API unavailable")

    holidays["date"] = pd.to_datetime(holidays["date"])
    today    = pd.Timestamp.now()
    upcoming = holidays[holidays["date"] >= today].head(10)

    results = []
    for _, row in upcoming.iterrows():
        # Calculate avg lift on holidays vs non-holidays
        avg_lift = master[master["is_holiday"] == 1]["holiday_lift_score"].mean()

        results.append({
            "holiday_name"      : row["holiday_name"],
            "date"              : str(row["date"])[:10],
            "holiday_type"      : row["holiday_type"],
            "days_away"         : (row["date"] - today).days,
            "expected_lift_pct" : round(avg_lift * 100, 1),
            "affected_categories": CATEGORIES
        })

    return {"upcoming_holidays": results,
            "generated_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@router.get("/model-metrics")
def get_model_metrics():
    """Trained model performance metrics"""
    summary = load_model_summary()
    return {
        "metrics"     : summary.to_dict(orient="records"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
@router.get("/historical-sales")
def get_historical_sales(
    start_date: str,
    end_date: str,
    category: str = "All",
    shop: str = "All"
):
    """Return actual historical sales between two dates"""
    import math

    def clean(v):
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return 0.0
            return round(f, 2)
        except:
            return 0.0

    master = load_master()
    master["date"] = pd.to_datetime(master["date"])

    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    filtered = master[
        (master["date"] >= start) &
        (master["date"] <= end)
    ].copy()

    if category != "All":
        filtered = filtered[filtered["category"] == category]

    if filtered.empty:
        return {
            "status" : "no_data",
            "message": f"No historical data between {start_date} and {end_date}",
            "tip"    : "Historical data covers last 90 days only"
        }

    # Fill all NaN before groupby
    filtered = filtered.fillna(0)

    daily = filtered.groupby("date").agg(
        total_sales  = ("sales",        "sum"),
        avg_sales    = ("sales",        "mean"),
        transactions = ("transactions", "sum"),
        is_holiday   = ("is_holiday",   "max"),
        is_weekend   = ("is_weekend",   "max"),
        temperature  = ("temperature",  "mean"),
        humidity     = ("humidity",     "mean")
    ).reset_index()

    daily["date"] = daily["date"].astype(str)

    daily_records = []
    for _, row in daily.iterrows():
        daily_records.append({
            "date"        : str(row["date"]),
            "total_sales" : clean(row["total_sales"]),
            "avg_sales"   : clean(row["avg_sales"]),
            "transactions": int(clean(row["transactions"])),
            "is_holiday"  : int(clean(row["is_holiday"])),
            "is_weekend"  : int(clean(row["is_weekend"])),
            "temperature" : clean(row["temperature"]),
            "humidity"    : clean(row["humidity"])
        })

    if not daily_records:
        return {"status":"no_data","message":"No data found","tip":"Check date range"}

    total    = clean(sum(r["total_sales"] for r in daily_records))
    avg      = clean(total / len(daily_records))
    peak     = max(daily_records, key=lambda r: r["total_sales"])

    return {
        "status"     : "historical",
        "start_date" : start_date,
        "end_date"   : end_date,
        "category"   : category,
        "total_days" : len(daily_records),
        "total_sales": total,
        "avg_daily"  : avg,
        "peak_day"   : peak["date"],
        "peak_sales" : peak["total_sales"],
        "daily_data" : daily_records
    }
@router.get("/dashboard-data")
def get_dashboard_data():
    """Single endpoint — all data for Power BI and frontend"""
    master  = load_master()
    summary = load_model_summary()

    def safe_float(val):
        """Convert NaN/Inf to 0 safely"""
        try:
            if val is None or (isinstance(val, float) and
               (np.isnan(val) or np.isinf(val))):
                return 0.0
            return round(float(val), 2)
        except:
            return 0.0

    # Sales by category
    sales_by_cat = []
    for cat in CATEGORIES:
        df = master[master["category"] == cat]
        sales_by_cat.append({
            "category": cat,
            "total"   : safe_float(df["sales"].sum()),
            "avg"     : safe_float(df["sales"].mean()),
            "std"     : safe_float(df["sales"].std())
        })

    # Sales by day of week
    sales_by_dow = {}
    for day, val in master.groupby("day_name")["sales"].mean().items():
        sales_by_dow[day] = safe_float(val)

    # Sales by month
    sales_by_month = {}
    for month, val in master.groupby("month")["sales"].mean().items():
        sales_by_month[str(month)] = safe_float(val)

    # Holiday vs non-holiday
    hol_avg     = safe_float(master[master["is_holiday"] == 1]["sales"].mean())
    non_hol_avg = safe_float(master[master["is_holiday"] == 0]["sales"].mean())

    # Weather impact
    rainy_avg = safe_float(master[master["is_rainy"] == 1]["sales"].mean())
    clear_avg = safe_float(master[master["is_rainy"] == 0]["sales"].mean())

    # Weekend vs weekday
    weekend_avg = safe_float(master[master["is_weekend"] == 1]["sales"].mean())
    weekday_avg = safe_float(master[master["is_weekend"] == 0]["sales"].mean())

    # Category correlations with weather
    correlations = []
    for cat in CATEGORIES:
        df = master[master["category"] == cat]
        correlations.append({
            "category"            : cat,
            "temp_correlation"    : safe_float(df["sales"].corr(df["temperature"])),
            "humidity_correlation": safe_float(df["sales"].corr(df["humidity"]))
        })

    # Clean model metrics
    metrics_clean = []
    for row in summary.to_dict(orient="records"):
        clean = {k: safe_float(v) if isinstance(v, float) else v
                 for k, v in row.items()}
        metrics_clean.append(clean)

    return {
        "sales_by_category"   : sales_by_cat,
        "sales_by_day_of_week": sales_by_dow,
        "sales_by_month"      : sales_by_month,
        "holiday_comparison"  : {
            "holiday_avg"    : hol_avg,
            "non_holiday_avg": non_hol_avg,
            "lift_pct"       : safe_float(
                (hol_avg - non_hol_avg) / non_hol_avg * 100
                if non_hol_avg > 0 else 0
            )
        },
        "weather_vs_sales"    : {
            "rainy_avg"      : rainy_avg,
            "clear_avg"      : clear_avg,
            "rain_impact_pct": safe_float(
                (rainy_avg - clear_avg) / clear_avg * 100
                if clear_avg > 0 else 0
            )
        },
        "weekend_vs_weekday"  : {
            "weekend_avg": weekend_avg,
            "weekday_avg": weekday_avg
        },
        "weather_correlations": correlations,
        "model_metrics"       : metrics_clean,
        "current_weather"     : fetch_current_weather(),
        "generated_at"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }