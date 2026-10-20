# Conformal forecast panels

The Forecast visualization and analysis page has parallel Point forecasts and
Conformal prediction tabs. Existing target, zone and date controls apply to both.
In the conformal tab, choose models, 80/90/95% coverage levels, and one or two
plots per row. Each model has its own forecast, actuals and nested interval bands.
The coverage table scores only displayed hours with both actuals and bands.

Calibration uses absolute errors separately for each model, zone, target and
six-hour horizon bucket. For every forecast origin, use at most 60 earlier,
fully observed 24-hour runs and require at least 20. A run becomes eligible
only after its final hourly delivery has ended, plus a two-hour reporting lag.
Neither its own outcomes nor future outcomes enter a forecast's calibration.
Original actual publication and revision vintages are unavailable: historical
calibration uses current stored actuals under that lag assumption. Retrospective
mock forecasts retain their existing provenance and are included.

For n eligible residuals, the radius is sorted residual number
ceil((n+1)*coverage). If that rank exceeds n, no finite band is displayed.
Bounds are forecast +/- radius, with lower bounds clipped at zero. These are
pointwise prediction intervals, not simultaneous 24-hour trajectory bands.
Time-series dependence and distribution changes can cause coverage to depart
from its nominal target; inspect observed coverage and width together.
See [Zaffran et al., 2022](https://proceedings.mlr.press/v162/zaffran22a.html)
for discussion of conformal calibration for dependent time series.

The daily forecast, outcome refresh and recovery artifact builders automatically
regenerate data/dashboard/conformal_forecasts.parquet, conformal_coverage.csv
and conformal_status.json. Precomputation avoids calibrating the entire archive
on each UI rerun. Only the selected target/zone is loaded into the panels.
