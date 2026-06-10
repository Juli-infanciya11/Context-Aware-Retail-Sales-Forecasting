from pydantic import BaseModel
from typing import Optional, List

class WeatherResponse(BaseModel):
    date: str
    city: str
    temperature: float
    humidity: int
    weather_condition: str
    description: str
    wind_speed: float
    feels_like: float
    fetched_at: str

class ForecastRequest(BaseModel):
    category: Optional[str] = "Electronics"
    days: Optional[int] = 30

class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float
    lower_bound: float
    upper_bound: float
    category: str

class ForecastResponse(BaseModel):
    category: str
    forecast_days: int
    forecasts: List[ForecastPoint]
    model_r2: float
    generated_at: str

class HolidayImpactResponse(BaseModel):
    holiday_name: str
    date: str
    holiday_type: str
    expected_lift_pct: float
    affected_categories: List[str]

class SalesSummaryResponse(BaseModel):
    category: str
    total_sales: float
    avg_daily_sales: float
    best_day: str
    worst_day: str
    weather_correlation: float
    holiday_lift_avg: float

class ModelMetrics(BaseModel):
    category: str
    xgb_r2: float
    xgb_mae: float
    sarima_r2: float
    sarima_mae: float
    ensemble_r2: float
    ensemble_mae: float