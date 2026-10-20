"""Optional HTTP API for a latest-value TRM forecast."""
from pathlib import Path
from trm_forecast.data.ingestion import load_trm
from trm_forecast.models.baseline import forecast_last

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover
    FastAPI = None


def create_app(data_path="data/raw/trm.csv"):
    if FastAPI is None:
        raise ImportError("install the 'app' extra to use the API")
    app = FastAPI(title="TRM Forecast API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/forecast")
    def forecast(horizon: int = 5):
        if horizon < 1 or horizon > 10:
            raise HTTPException(400, "horizon must be between 1 and 10")
        if not Path(data_path).exists():
            raise HTTPException(404, "TRM data file not found")
        data = load_trm(data_path)
        result = forecast_last(data["trm"], horizon)
        return {"forecast": [{"date": date.isoformat(), "trm": value} for date, value in result.items()]}

    return app


app = create_app() if FastAPI is not None else None
