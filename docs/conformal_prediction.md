# Probabilistic forecast panels

The Forecast visualization and analysis page has parallel Point forecasts and
Probabilistic forecasts tabs. Existing target, zone and date controls apply to both.
In the probabilistic tab, choose models and one or two plots per row.
The displayed band has 80% nominal coverage, matching the interval coverage
of Elia’s P10–P90 wind and solar forecasts. The conformal bounds are symmetric
absolute-error intervals, not separately calibrated P10 and P90 quantiles.
See [Elia wind forecast fields](https://opendata.elia.be/explore/dataset/ods086/information/)
and [Elia solar forecast fields](https://opendata.elia.be/explore/dataset/ods087/information/). Each model has its own forecast, actuals and an 80% interval band.
Coverage diagnostics remain available in the generated CSV artifact.

Calibration uses absolute errors separately for each model, zone, target and
six-hour horizon bucket. For every forecast origin, use at most 60 earlier,
fully observed complete 29/30/31-hour runs and require at least 20. A run becomes eligible
only after its final hourly delivery has ended, plus a two-hour reporting lag.
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
and conformal_status.json. Precomputation avoids calibrating the entire archive
on each UI rerun. Only the selected target/zone is loaded into the panels.
