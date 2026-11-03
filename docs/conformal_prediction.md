# Probabilistic forecast panels

The Forecast visualization and analysis page has parallel Point forecasts and
Probabilistic forecasts views. Existing target, zone and date controls apply to both.
In the probabilistic view, choose models and one or two plots per row.
The displayed interval has 80% nominal coverage, matching the interval coverage
of Elia’s P10–P90 wind and solar forecasts. The conformal bounds are symmetric
absolute-error intervals, not separately calibrated P10 and P90 quantiles.
See [Elia wind forecast fields](https://opendata.elia.be/explore/dataset/ods086/information/)
and [Elia solar forecast fields](https://opendata.elia.be/explore/dataset/ods087/information/). Each model has its own forecast, actuals and an 80% prediction interval.
Coverage diagnostics remain available in the generated CSV artifact.

## Probabilistic leaderboard page

The sidebar's **Probabilistic forecasts** page (`/probabilistic_forecasts`) has
the same task selector, model-family rows, bidding-zone columns and model order
as the deterministic Leaderboard. It displays three separate tables:

1. **Winkler score (MW):** mean 80% interval score; lower is better.
2. **Mean interval width (MW).**
3. **Coverage (%):** nominal target 80%.

All three use identical complete, observed, calibrated runs across the compared
models within each area. Scores are precomputed in
`data/dashboard/probabilistic_leaderboard.parquet`, so the page loads the small
selected-task snapshot. The deterministic page does not load interval results.
Forecast visualizations remain on More detailed insights.

For the 80% interval `[L, U]`, each hourly Winkler score is
`(U - L) + 10 * max(L - actual, 0) + 10 * max(actual - U, 0)`.
The table averages these hourly scores on the shared sample. The multiplier is
`2 / (1 - coverage)`; an observation exactly on either bound incurs no miss
penalty. Width and Winkler score have MW units; compare models within a column,
rather than averaging scores across differently sized systems. This follows
[Forecasting: Principles and Practice — interval accuracy](https://otexts.com/fpp3/distaccuracy.html).

Calibration uses absolute errors separately for each model, zone, target and
six-hour horizon bucket. For every forecast origin, use at most 60 earlier,
fully observed complete 29/30/31-hour runs and require at least 20. A run becomes eligible
only after its final hourly delivery has ended, plus a two-hour reporting lag.
The five buckets are hours 1–6, 7–12, 13–18, 19–24 and 25–30; hour 31
shares the final bucket. All comparisons require the exact complete UTC grid
defined by the [shared delivery-window contract](FORECAST_WINDOW_MIGRATION.md#operator-and-adapter-contract).
Neither its own outcomes nor future outcomes enter a forecast's calibration.
Original actual publication and revision vintages are unavailable: historical
calibration uses current stored actuals under that lag assumption. Retrospective
mock forecasts retain their existing provenance and are included.

For n eligible residuals, the radius is sorted residual number
ceil((n+1)*coverage). If that rank exceeds n, no finite band is displayed.
Bounds are forecast +/- radius, with lower bounds clipped at zero. These are
pointwise prediction intervals, not simultaneous full-window trajectory bands.
Time-series dependence and distribution changes can cause coverage to depart
from its nominal target; inspect observed coverage and width together.
See [Zaffran et al., 2022](https://proceedings.mlr.press/v162/zaffran22a.html)
for discussion of conformal calibration for dependent time series.

The daily forecast, outcome refresh and recovery artifact builders automatically
regenerate data/dashboard/conformal_forecasts.parquet, conformal_coverage.csv
and conformal_status.json, plus the probability score snapshot. Precomputation avoids calibrating the entire archive
on each UI rerun. Only the selected target/zone is loaded into the panels.
