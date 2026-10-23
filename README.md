# TRM Forecasting & Risk

Forecast Colombia's Representative Market Exchange Rate (TRM, USD/COP) over a 1–10 business-day horizon and translate forecasts into exposure metrics for international trade operations.

## Quick start

```bash
pip install -e '.[dev]'
pytest
```

The ingestion module downloads the `ceyp-9c7c` TRM dataset without requiring
API credentials. The reproducible full-history command appears below.

The source endpoint and field names are configured in `configs/base.yaml`. The downloader stores the combined raw records and request metadata in `RawDownload`; normalization is a separate operation and produces `date`, `trm`, `source`, and `retrieved_at`. Socrata requests use the configured date filters, `$limit`, `$offset`, and deterministic `$order`. Pages are combined until a short or empty page is returned, with a configurable `max_pages` guard.

## Temporal evaluation and persistence baseline

The first forecasting experiment uses the reusable temporal protocol in
`src/trm_forecast/evaluation/protocol.py` and the last-observation persistence
baseline in `src/trm_forecast/evaluation/persistence.py`. Horizons count
subsequent TRM observations, not calendar days. The initial configurable split
is:

- Training: 2008-01-01 through 2022-12-31.
- Validation: 2023-01-01 through 2024-12-31.
- Test: 2025-01-01 through the latest eligible observation.
- As-of cutoff: 2026-10-19.

The cutoff excludes the source record dated 2026-10-20. It does not use
`retrieved_at` as a historical publication vintage, because original source
vintages are unavailable. These boundaries are a reproducible initial proposal,
not an assertion that the split is optimal. Change them in the `evaluation`
section of `configs/base.yaml`.

Run the persistence experiment with:

```bash
./.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/04_persistence_baseline.ipynb --output /tmp/trm_persistence_baseline.ipynb
```

The experiment reports partition counts and date ranges, predictions for each
eligible origin and horizon, and MAE/RMSE grouped by partition and horizon. It
does not include MASE because no separate, correctly defined training-scale
denominator has been selected yet. A lower error than persistence is necessary
for a more complex model to be interesting, but it is not sufficient evidence
of business value.

## Business-value evaluation

`src/trm_forecast/business/evaluation.py` evaluates timing decisions from the
same origin/horizon forecast table. For each origin and model, the importer
selects the horizon with the lowest predicted TRM and the exporter selects the
horizon with the highest predicted TRM. Horizon 1 is the execution baseline.
Selection uses predictions only; actual TRM values are read only after the
horizon has been selected.

The evaluator requires an explicit positive `usd_amount`, for example:

```python
from trm_forecast.business.evaluation import evaluate_business_value

decisions, summary = evaluate_business_value(
    predictions, usd_amount=100_000, horizons=tuple(range(1, 11))
)
```

For an importer, gross COP benefit is `USD × (actual_TRM_h1 -
actual_TRM_selected)`. For an exporter it is `USD × (actual_TRM_selected -
actual_TRM_h1)`. Positive values improve on executing at horizon 1; negative
values are worse outcomes. Decision rows include the origin, model, operation,
selected horizon, predicted and realized TRM at selection, realized TRM at
horizon 1, and gross COP benefit. Aggregates report mean and median benefit,
percentage of positive benefits, valid decision count, and mean selected
horizon by partition, model, and operation.

Strict mode raises for duplicate origin/model/horizon forecasts, missing
horizons, or non-finite predictions/actuals. With `strict=False`, invalid
groups are excluded and `validate_forecast_grid` reports the issue type; no
forecast or realization is fabricated. Ties select the lowest horizon after
sorting, making horizon 1 the decision for a flat forecast.

Model or decision-rule choices must be made on validation data. The test
partition is reserved for the final frozen comparison. The included
`notebooks/08_business_value_evaluation.ipynb` compares persistence,
ARIMA(0,1,0), ARIMA(0,1,1), and ARIMA(0,1,0) with drift on equivalent
level-forecast origins and horizons. For integrated models, the drift variant
uses statsmodels `trend="t"`, which represents a constant in first
differences; a literal constant is invalid when `d=1`. The drift variant is
reported as a predefined experiment, not selected using test results. GARCH
and regime-switching outputs forecast volatility rather than TRM levels, so
they are not treated as timing models in this comparison.
Business value is gross only: it excludes spreads, fees, hedging costs, taxes,
liquidity, execution uncertainty, and the possibility that a future forecast
is unavailable or cannot be executed.

## TRM-derived Ridge feature experiment

`src/trm_forecast/evaluation/ridge.py` evaluates direct horizon-specific Ridge
regressions using the feature builder in
`src/trm_forecast/features/build_features.py`. The feature set contains the
observed origin TRM level, lagged TRM levels, the log return through the
origin, and rolling means/standard deviations computed from
`trm.shift(1)`, plus day of week. Ridge standardization and coefficients are
fit on training rows only. Alpha is selected on validation; the selected alpha
is then refit on train plus validation before the frozen test evaluation.

The experiment is documented in
`notebooks/09_exogenous_features_ridge.ipynb` and its executed HTML report.
No external macroeconomic variables are included in the primary experiment:
the repository has no historical external dataset with publication or
availability timestamps, so joining US rates, dollar indices, Colombian policy
rates, or inflation by economic period would not be leakage-safe. Such data
require vintage-aware sources before they can be added.

The robustness audit is documented in
`reports/10_ridge_economic_robustness.md`. It uses the frozen validation
choice `alpha = 0.1` and common complete origin/horizon groups to report
benefit percentiles, positive/zero/negative proportions, horizon-selection
distributions, concentration of positive outcomes, quarterly stability,
matched-origin comparisons, and USD-notional sensitivity. These summaries are
descriptive; they do not establish economic superiority and exclude execution
costs and other real-world frictions.

The consolidated technical source of truth is in
`reports/experiment_inventory.md` and `reports/master_results.md`, with
machine-readable level, volatility, and business-value tables in the adjacent
CSV files. These tables were regenerated from the current evaluation modules;
HTML exports are not treated as authoritative when their execution history is
uncertain.

## ARIMA experiment

`src/trm_forecast/models/arima.py` provides a configurable statsmodels ARIMA
wrapper and train-only ADF/KPSS diagnostics. `src/trm_forecast/evaluation/arima.py`
refits model parameters on the configured schedule using observations available
up to each origin, and updates the fitted state from current-origin history
between refits. Failed fits remain in the prediction output and are counted
explicitly. Candidate orders are selected by validation MAE only; the selected
order is then frozen for the test comparison against persistence on the shared
origin/target grid.

Install the optional statistical-model dependency and execute the experiment:

```bash
pip install -e '.[dev,models]'
./.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/05_arima_forecast.ipynb --output /tmp/trm_arima_forecast.ipynb
```

The candidate orders, minimum history, optimizer limit, and refit policy are
configured under `arima` in `configs/base.yaml`. Stationarity tests are
diagnostics, not proof of a model choice; visual fit and historical superiority
over persistence do not establish performance for future 1–10 business-day
operations.

## GARCH volatility experiment

`src/trm_forecast/models/garch.py` and `src/trm_forecast/evaluation/garch.py`
implement a leakage-aware GARCH(1,1) conditional-volatility experiment. The
target is 100 times the log return, not the TRM level. The model uses a
constant conditional mean and GARCH(1,1) variance, refitting every 10 origins
and updating its state with current-origin returns between refits. The existing
optional `models` extra supplies the `arch` dependency:

```bash
pip install -e '.[dev,models]'
./.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/06_garch_volatility.ipynb --output /tmp/trm_garch_volatility.ipynb
```

At horizon `h`, the realized volatility proxy is the squared cumulative scaled
log return from the origin to the h-th subsequent observation. It is a noisy
proxy for latent integrated variance, not the true conditional variance.
Forecast variance is the sum of the h recursive one-step conditional
variances and is scored with QLIKE; these scores must not be compared with
level MAE/RMSE. The optional 95% TRM-level intervals are a plug-in normal
approximation based on that summed variance, not an exact finite-horizon GARCH
predictive distribution. Coverage uncertainty uses a moving-block bootstrap
with block length 10 to reflect overlapping forecast windows.

On the local full-history data, training returns showed strong ARCH evidence
(Engle LM p-value `4.18e-113`; squared-return Ljung–Box p-values below
`4e-223`). All 4,665 validation and 4,175 test forecasts succeeded. Mean
validation QLIKE across horizons was `2.386`, with mean coverage `0.920` and
mean interval width `323.1` COP/USD. Mean test QLIKE was `2.095`, coverage
`0.932`, and mean interval width `260.0` COP/USD. The untuned expanding
historical-variance benchmark had mean QLIKE `2.346` on validation and `2.092`
on test, so GARCH did not improve this volatility loss overall. These are
volatility and interval results; they do not demonstrate improved TRM level
forecasting.
The return proxy is not a directly observed latent volatility measure, and the
normality/uncorrelated-innovation assumptions may be inadequate during market
stress.

## Markov regime-switching experiment

`src/trm_forecast/models/regime_switching.py` and
`src/trm_forecast/evaluation/regime_switching.py` implement one parsimonious
two-state Markov-switching model for the same scaled log returns. It has a
common constant conditional mean, regime-specific variances, and a constant
two-state transition matrix. At each origin, the model is fitted using only
returns available through that origin. Between refits, the estimated
parameters are held fixed and the filtered state probabilities are recomputed
from the newly observed returns. Smoothed probabilities are never used for
forecasting.

Run the experiment with:

```bash
./.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/07_regime_switching.ipynb --output /tmp/trm_regime_switching.ipynb
```

The model produced 4,665 successful validation forecasts and 4,175 successful
test forecasts. Mean QLIKE was `2.386` on validation and `2.085` on test,
compared with the expanding historical-variance benchmark's `2.346` and
`2.092`, respectively. GARCH scored `2.386` and `2.095` on the same
validation/test evaluations. The regime model therefore did not improve
validation QLIKE, although it was slightly better on this test period. That
test-period difference is not sufficient evidence of general superiority.

The regime-switching model is a volatility experiment, not a TRM-level
forecasting replacement. Its finite-horizon variance uses filtered regime
probabilities propagated through the transition matrix. The resulting return
distribution is a regime mixture, so no normal level interval is reported.
State-label identification, local likelihood maxima, and short regime
episodes remain limitations.

## Layout

- `src/trm_forecast/data`: ingestion and schema validation.
- `src/trm_forecast/data/audit.py`: non-destructive historical quality audit.
- `src/trm_forecast/features`: lag, return, rolling-volatility, and calendar features.
- `src/trm_forecast/models`: baseline and optional statistical/deep-learning model adapters.
- `src/trm_forecast/evaluation`: time-series backtesting and error metrics.
- `src/trm_forecast/business`: exposure, scenario, and timing-value calculations.
- `src/trm_forecast/pipeline`: an end-to-end daily forecast runner.
- `app`: small FastAPI and Streamlit entry points.

The project is deliberately leakage-aware: features are shifted before being used for prediction, and backtests only train on observations available before each forecast origin.

## Data source

The source is the [TRM dataset `ceyp-9c7c` on Datos Abiertos Colombia](https://www.datos.gov.co/Econom-a-y-Finanzas/TRM/ceyp-9c7c), supplied by the Superintendencia Financiera de Colombia. Its verified API fields are `valor`, `vigenciadesde`, and `vigenciahasta`; this implementation uses `vigenciadesde` as `date` and `valor` as `trm`. Socrata SoQL date filters are sent when `start_date` and/or `end_date` are provided.

Validation is strict: invalid dates, missing/non-numeric/non-finite/non-positive TRM values, and duplicate observation dates raise errors. No observations are imputed, removed, or silently deduplicated. Weekends and Colombian holidays are not required; actual source dates are retained.

Tests use injected HTTP responses and never depend on the API. The raw and
interim datasets generated by live retrieval are ignored by Git and remain
local execution artifacts.

To retrieve an explicit exploratory range and save the normalized result:

```bash
./.venv/bin/python -c "import json; from pathlib import Path; from trm_forecast.data.ingestion import download_trm, normalize_download; raw = download_trm(start_date='2008-01-01', end_date=None); Path('data/raw/trm_full_records.json').write_text(json.dumps(raw.records, ensure_ascii=False, indent=2)); normalized = normalize_download(raw); normalized.to_csv('data/interim/trm_normalized.csv', index=False); print('raw records:', len(raw.records)); print('normalized records:', len(normalized)); print('latest:', normalized['date'].max().date())"
```

The start date is explicit while `end_date=None` lets the source determine its
latest available observation. This is not a final training, validation, or
test-period decision.

## Historical audit and exploratory analysis

Run the quality audit against the normalized local file:

```bash
./.venv/bin/python -c "import pandas as pd; from trm_forecast.data.audit import audit_trm; print(audit_trm(pd.read_csv('data/interim/trm_normalized.csv')).to_dict())"
```

The audit is non-destructive. It reports valid observation counts, date coverage,
duplicates, invalid dates and values, chronological ordering, and calendar and
generic Monday-Friday intervals. A candidate gap is not proof of a missing TRM:
weekends, Colombian public holidays, publication conventions, and source outages
are not inferred. No values are filled, interpolated, deduplicated, or removed.

Execute the notebooks from the repository root with Jupyter installed:

```bash
./.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/01_data_audit.ipynb --output /tmp/trm_data_audit.ipynb
./.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/02_exploratory_analysis.ipynb --output /tmp/trm_exploratory_analysis.ipynb
```

The regenerated local file is `data/interim/trm_normalized.csv`. The latest
full-history run contains 4,456 observations from 2008-01-03 through
2026-10-20, with columns `date`, `trm`, `source`, and `retrieved_at`. The raw
response is preserved locally as `data/raw/trm_full_records.json`; both data
files are ignored by Git. The source includes dates such as Saturdays because
TRM validity dates are source observations, not an assumed Monday-Friday
schedule. This exploratory coverage is not a final model-period decision.

Run all tests with:

```bash
./.venv/bin/python -m pytest
```
