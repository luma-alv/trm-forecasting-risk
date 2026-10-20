# TRM Forecasting & Risk

Forecast Colombia's Representative Market Exchange Rate (TRM, USD/COP) over a 1–10 business-day horizon and translate forecasts into exposure metrics for international trade operations.

## Quick start

```bash
pip install -e '.[dev]'
pytest
python -m trm_forecast.pipeline.daily_run --input data/raw/trm.csv --output data/processed/forecast.csv
```

The input CSV must contain a date column (`date`, `fecha`, or `Date`) and a TRM column (`trm`, `value`, or `valor`). Dates are normalized to business days and values are validated as positive.

## Layout

- `src/trm_forecast/data`: ingestion and schema validation.
- `src/trm_forecast/features`: lag, return, rolling-volatility, and calendar features.
- `src/trm_forecast/models`: baseline and optional statistical/deep-learning model adapters.
- `src/trm_forecast/evaluation`: time-series backtesting and error metrics.
- `src/trm_forecast/business`: exposure and scenario calculations.
- `src/trm_forecast/pipeline`: an end-to-end daily forecast runner.
- `app`: small FastAPI and Streamlit entry points.

The project is deliberately leakage-aware: features are shifted before being used for prediction, and backtests only train on observations available before each forecast origin.

## Data source

Place an official or licensed TRM export in `data/raw/`. The ingestion layer is source-agnostic so a Banco de la República/Superintendencia Financiera download can be used without changing model code. Do not commit confidential exposure data.
