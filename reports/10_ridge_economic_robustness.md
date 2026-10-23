# Ridge economic-value robustness audit

This audit uses the existing forecast generators and business evaluator. It
does not fit a new model, change the importer/exporter rule, or use test data
to select a model or hyperparameter. Ridge uses the alpha selected previously
on validation (`alpha = 0.1`), frozen for this audit. All models are compared
on the same complete origin/horizon groups: 462 validation origins and 413
test origins, with two operation decisions per origin.

## Findings

For USD 100,000, the decision-level results are:

| Partition | Model | Operation | Mean COP | Median COP | P05–P95 COP | Positive / zero / negative | Valid |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | Ridge | importer | 832,641 | 0 | -6,378,250 to 11,603,000 | 32.7% / 48.1% / 19.3% | 462 |
| validation | Ridge | exporter | -236,366 | -228,000 | -18,390,800 to 16,314,750 | 45.2% / 4.1% / 50.6% | 462 |
| test | Ridge | importer | 811,603 | 0 | -7,787,200 to 9,214,200 | 37.8% / 38.0% / 24.2% | 413 |
| test | Ridge | exporter | -1,653,838 | -1,966,000 | -13,378,400 to 13,718,200 | 36.3% / 2.9% / 60.8% | 413 |

Ridge selected later than horizon 1 for 51.9% of validation and 62.0% of
test importer decisions, and for 95.9% of validation and 97.1% of test
exporter decisions. The mean test result is therefore driven by a materially
different decision rule from the persistence/ARIMA level forecasts, not by a
change to the baseline.

Positive-outcome concentration is moderate rather than negligible. In test,
the largest 1% of positive Ridge outcomes account for 6.3% of positive
importer benefit and 5.5% of positive exporter benefit; the corresponding
10% shares are 29.6% and 30.4%. Concentration is undefined for models with no
positive outcomes, rather than being reported as zero.

Matched-origin comparisons against persistence, ARIMA(0,1,0), and
ARIMA(0,1,1) are identical because those models select horizon 1 on every
complete origin. Against that common reference, Ridge's mean test difference
is +811,603 COP for importers and -1,653,838 COP for exporters. Ridge is
better than the reference on 37.8% of test importer decisions and 36.3% of
test exporter decisions; the remaining decisions are ties or worse outcomes.

Against ARIMA(0,1,0) with drift, Ridge's mean test difference is +811,603 COP
for importers and +808,937 COP for exporters. However, Ridge is better on only
37.8% and 25.4% of those matched decisions, respectively; 62.0% of exporter
decisions are ties because the drift model selects horizon 10 while Ridge
also selects horizon 10.

## Amount sensitivity

Decision selection is independent of the USD notional, so changing the amount
scales every gross COP benefit linearly and leaves positive/zero/negative
percentages unchanged. For example, test Ridge importer mean benefit is
81,160 COP at USD 10,000, 405,801 COP at USD 50,000, 811,603 COP at USD
100,000, and 4,058,015 COP at USD 500,000. This is not a risk-adjusted
profitability result: spreads, fees, taxes, liquidity, execution failure,
credit, and opportunity costs are excluded.

## Stability and interpretation

Quarterly summaries show that Ridge importer mean benefit changes sign in the
test period (positive through 2026 Q2, negative in 2026 Q3, with only two
decisions in 2026 Q4). Ridge exporter mean benefit is negative in every test
quarter. Validation is similarly heterogeneous: importer means are negative
in 2023 Q2 and 2024 Q3, while exporter means are positive in 2023 Q3 and
2024 Q2–Q4. These results do not support a claim of stable economic
superiority.

The audit is implemented in
`src/trm_forecast/business/audit.py`, with focused tests in
`tests/test_business_audit.py`. It reports percentiles, sign proportions,
selected-horizon distributions, top-positive-outcome concentration,
non-overlapping quarterly stability, matched-origin comparisons, and notional
sensitivity. The existing `notebooks/09_exogenous_features_ridge.ipynb` and
HTML report remain the source for Ridge alpha selection and predictive
metrics; this audit consumes the same protocol and frozen alpha.

## Limitations

The analysis is retrospective and assumes that all selected future horizons
could be executed at their observed TRM. It does not model transaction costs,
execution timing, liquidity, limits, or forecast revision. The common-origin
filter excludes incomplete horizon-10 groups and therefore describes the
complete-grid sample, not every available forecast. Economic results are
descriptive comparisons, not evidence that Ridge is economically superior.
