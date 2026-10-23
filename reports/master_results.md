# Master TRM forecasting results

This is the consolidated technical source of truth for the current project
state. Horizon-level level-forecast metrics are in
[`master_predictive_metrics.csv`](master_predictive_metrics.csv); volatility
metrics are kept separately in
[`master_volatility_metrics.csv`](master_volatility_metrics.csv); business
results are in [`master_business_value.csv`](master_business_value.csv).

## Methodology and data

The source is the normalized Colombia TRM series with 4,456 valid unique
observations from 2008-01-03 to 2026-10-20. The configured as-of cutoff is
2026-10-19, excluding one post-cutoff observation. The unchanged partitions
are train 2008–2022, validation 2023–2024, and test 2025–2026-10-19.
Horizons 1–10 count later observed TRM records, not calendar days.

Level forecasts use the common evaluation target grid. Incomplete terminal
horizons are represented by decreasing valid counts; no values are fabricated.
Models and selection rules are not tuned on test. Ridge alpha `0.1` was
selected using validation only, then frozen for test. Its features are
TRM-derived: current origin level, lags 1/2/5/10, log return, past-only
rolling statistics, and day of week. Scaling is fit on training rows only.
External macro predictors were excluded because the repository lacks reliable
historical publication/availability timestamps for leakage-safe joins.

## Predictive results

Mean across horizons, with total valid forecasts in parentheses:

| Partition | Persistence | ARIMA(0,1,0) | ARIMA(0,1,1) | ARIMA(0,1,0) drift | Ridge |
|---|---:|---:|---:|---:|---:|
| Validation MAE | 67.448 (4,665) | 67.448 (4,665) | 67.505 (4,665) | 67.847 (4,665) | 68.536 (4,665) |
| Validation RMSE | 87.532 | 87.532 | 87.438 | 87.951 | 88.619 |
| Test MAE | 53.646 (4,175) | 53.646 (4,175) | 53.690 (4,175) | 54.399 (4,175) | 54.876 (4,175) |
| Test RMSE | 67.743 | 67.743 | 67.811 | 68.384 | 68.928 |

Persistence and ARIMA(0,1,0) are identical in this implementation. Ridge is
not the lowest-error model in either partition under the frozen alpha. The CSV
contains the exact MAE, RMSE, valid count, excluded terminal count, and
provenance for every model/partition/horizon row.

## Volatility results

QLIKE is not comparable with level MAE/RMSE. Mean QLIKE across horizons is:

| Partition | Historical variance | GARCH(1,1) | Markov switching, 2 state |
|---|---:|---:|---:|
| Validation | 2.345826 | 2.385991 | 2.386324 |
| Test | 2.092427 | 2.094501 | 2.084390 |

The separate CSV also contains horizon-level QLIKE, GARCH coverage and mean
interval width. These models forecast volatility, not TRM level timing.

## Business value

Business results use USD 100,000 and the established rules: importer selects
the minimum predicted TRM, exporter the maximum, and horizon 1 is the control.
Only predictions select the horizon; realized TRM is used afterward. The
common complete-grid sample contains 462 validation and 413 test origins.
Each model/operation has 18 excluded decisions from the nine terminal
origin groups per partition.

| Partition | Model | Operation | Mean COP | Median COP | Positive % | Mean selected horizon | Later than 1 % |
|---|---|---|---:|---:|---:|---:|---:|
| Validation | Ridge | Importer | 832,641 | 0 | 32.7 | 2.95 | 51.9 |
| Validation | Ridge | Exporter | -236,366 | -228,000 | 45.2 | 8.27 | 95.9 |
| Test | Ridge | Importer | 811,603 | 0 | 37.8 | 3.34 | 62.0 |
| Test | Ridge | Exporter | -1,653,838 | -1,966,000 | 36.3 | 8.27 | 97.1 |

Persistence, ARIMA(0,1,0), and ARIMA(0,1,1) select horizon 1 for all complete
decisions, so their gross benefit is zero by construction. The drift model
selects horizon 10 for exporters and has negative mean test benefit. The Ridge
robustness audit shows material downside tails and quarterly instability;
these are gross theoretical COP comparisons, not net profit or guaranteed
savings. Spreads, fees, taxes, liquidity, credit, execution uncertainty, and
future forecast availability are excluded.

## Limitations and unresolved discrepancies

The project contains generated HTML artifacts with uneven execution evidence:
notebooks 01/02/04 contain outputs, notebook 09 has a recent successfully
executed HTML export, while notebooks 05–08 do not contain current execution
outputs in their `.ipynb` files. Consequently, the tables in this report are
source-code regenerations and should supersede unsupported stale exports.
The current worktree also contains user modifications and deleted HTML files;
they were not discarded or restored.

No Deep Learning experiment or external-macro experiment has been completed.
The remaining academic-report risk is artifact provenance: if exact historical
notebook outputs are required, the relevant notebooks should be re-executed
and their HTML exports regenerated from the current commit before submission.
