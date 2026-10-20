# TRM Forecasting & Risk

Forecast Colombia's Representative Market Exchange Rate (TRM, USD/COP) over a 1–10 business-day horizon and translate forecasts into exposure metrics for international trade operations.

## Quick start

```bash
pip install -e '.[dev]'
pytest
```

The first implemented module downloads the `ceyp-9c7c` TRM dataset without requiring API credentials:

```bash
python -c "from trm_forecast.data.ingestion import download_trm, normalize_download; raw = download_trm(start_date='2024-01-01', end_date='2024-01-31'); normalize_download(raw).to_csv('data/interim/trm_normalized.csv', index=False)"
```

The source endpoint and field names are configured in `configs/base.yaml`. The downloader stores raw records and request metadata; normalization is a separate operation and produces `date`, `trm`, `source`, and `retrieved_at`.

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

The source is the [TRM dataset `ceyp-9c7c` on Datos Abiertos Colombia](https://www.datos.gov.co/Econom-a-y-Finanzas/TRM/ceyp-9c7c), supplied by the Superintendencia Financiera de Colombia. Its verified API fields are `valor`, `vigenciadesde`, and `vigenciahasta`; this implementation uses `vigenciadesde` as `date` and `valor` as `trm`. Socrata SoQL date filters are sent when `start_date` and/or `end_date` are provided.

Validation is strict: invalid dates, missing/non-numeric/non-finite/non-positive TRM values, and duplicate observation dates raise errors. No observations are imputed, removed, or silently deduplicated. Weekends and Colombian holidays are not required; actual source dates are retained.

Known limitation: network access to the external API was unavailable in this execution environment, so live downloading could not be exercised here. Tests use injected HTTP responses and never depend on the API.
