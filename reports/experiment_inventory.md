# Experiment inventory and reproducibility status

Audit date: 2026-10-22. Branch: `feat/business-value-evaluation`.
Latest commit: `44b3963542cde0ea1d53b054576fbc99fd115166` (`test(business):
audit forecast horizons and economic evaluation`, 2026-10-20).

## Shared protocol

The configured cutoff is 2026-10-19; the normalized source contains 4,456
valid unique observations from 2008-01-03 through 2026-10-20, so one record is
excluded after the cutoff. Partitions are train 2008–2022, validation
2023–2024, and test 2025–2026-10-19. Horizons 1–10 are subsequent TRM
observations, not calendar days. The evaluation code does not impute missing
observations and uses only history available at each origin.

| Experiment | Purpose and specification | Source artifacts | Current execution/evidence | Limitations |
|---|---|---|---|---|
| Historical ingestion and audit | Download/normalize Colombia open-data TRM; check dates, values, duplicates, and gaps | `src/trm_forecast/data/ingestion.py`, `src/trm_forecast/data/audit.py`, `notebooks/01_data_audit.ipynb` | Notebook contains executed outputs; HTML export is deleted in the current worktree. Direct audit confirms 4,456 valid rows, no duplicate dates, missing values, or invalid TRM values. | `retrieved_at` is not a historical publication vintage; generic Monday–Friday gaps do not encode Colombian holidays. |
| Exploratory analysis | Describe level, return, distribution, and temporal behavior | `notebooks/02_exploratory_analysis.ipynb` | Notebook contains executed outputs; HTML export is deleted in the current worktree. | Descriptive only; no causal inference. |
| Persistence baseline | Last observed TRM at every future horizon | `src/trm_forecast/evaluation/persistence.py`, `notebooks/04_persistence_baseline.ipynb` | Notebook contains executed outputs; metrics were regenerated directly for the master CSV. | Baseline is intentionally simple and is the horizon-1 economic control. |
| ARIMA comparison | Expanding/refit ARIMA(0,1,0), ARIMA(0,1,1), and predefined drift variant (`trend="t"`) | `src/trm_forecast/evaluation/arima.py`, `src/trm_forecast/models/arima.py`, `notebooks/08_business_value_evaluation.ipynb` | Horizon-level metrics regenerated from current source/configuration. HTML exists, but notebook execution counts are empty and source/HTML equivalence cannot be assumed. | Drift is a predefined comparison, not test-selected; statsmodels warnings remain in tests. |
| GARCH(1,1) | Forecast cumulative-return variance and level intervals; score QLIKE | `src/trm_forecast/evaluation/garch.py`, `src/trm_forecast/models/garch.py`, `notebooks/06_garch_volatility.ipynb` | Horizon-level volatility metrics regenerated directly. Notebook has no execution outputs and no current HTML export, so notebook execution is not claimed. | QLIKE/coverage are volatility metrics and must not be compared with level MAE/RMSE. |
| Two-state Markov switching | Filtered regime-switching variance forecasts; score QLIKE | `src/trm_forecast/evaluation/regime_switching.py`, `src/trm_forecast/models/regime_switching.py`, `notebooks/07_regime_switching.ipynb` | Horizon-level volatility metrics regenerated directly. Notebook execution is not evidenced by current outputs. | Regime identification and local optimization are limitations; this is not a level forecast. |
| Business-value evaluation | Importer chooses minimum prediction, exporter maximum; compare realized selected TRM with horizon 1 | `src/trm_forecast/business/evaluation.py`, `notebooks/08_business_value_evaluation.ipynb` | Business CSV regenerated on common complete origins: 462 validation and 413 test origins. | Gross theoretical benefit excludes spread, fees, tax, liquidity, and execution constraints. |
| Ridge feature forecast | Direct horizon-specific Ridge using current TRM, lags, returns, past-only rolling features, and day-of-week | `src/trm_forecast/models/ridge.py`, `src/trm_forecast/evaluation/ridge.py`, `notebooks/09_exogenous_features_ridge.ipynb` | HTML export timestamp is 2026-10-22 18:23 and was successfully produced by `nbconvert --execute`; alpha 0.1 was selected on validation. Master metrics were independently regenerated. | Only TRM-derived features are used. No external macro variable has verified historical availability timestamps. |
| Ridge robustness audit | Distribution, concentration, quarterly stability, matched origins, and notional sensitivity | `src/trm_forecast/business/audit.py`, `tests/test_business_audit.py`, `reports/10_ridge_economic_robustness.md` | Report and focused tests are current; results were recomputed from the frozen pipeline. | Retrospective gross outcomes do not establish stable economic superiority. |

## Git and artifact status

At audit time, `git status --short` reported modified `README.md`,
`configs/base.yaml`, notebook 08, ARIMA source/tests; deleted tracked HTML
exports for notebooks 01 and 02; and untracked Ridge notebooks, source/tests,
audit code, and reports. These changes were preserved. No reset, stash,
commit, or push was performed. The deleted HTML files are user/worktree
changes and were not restored.

The current HTML exports are not uniformly authoritative: notebook 09 has
recent execution evidence, while notebooks 05–08 either have no executed
outputs in their `.ipynb` files or have HTML/source timestamp uncertainty.
The master tables therefore identify direct source-code regeneration as their
provenance instead of treating an old HTML file as proof of current results.

## Quality checks

The full test suite passed: 62 tests, with six existing statsmodels-related
warnings. Targeted Ruff checks for the modified Ridge, business-audit, and
feature files passed. No VaR logic or unrelated evaluation protocol was
changed.
