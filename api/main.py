from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Retail Sales Forecasting API",
    description="Context-Aware Retail Sales Forecasting using Weather & Holiday Analytics",
    version="1.0.0"
)

# Allow frontend and Power BI to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)