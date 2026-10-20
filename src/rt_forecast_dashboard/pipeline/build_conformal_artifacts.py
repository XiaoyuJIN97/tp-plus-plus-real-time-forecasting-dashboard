"""Precompute conformal bands without adding model runtimes to the dashboard."""
import json
from datetime import datetime, UTC
from rt_forecast_dashboard.conformal import (
    conformal_intervals, interval_diagnostics, LEVELS, WINDOW_RUNS, MIN_RUNS, REPORTING_LAG_HOURS,
)
from rt_forecast_dashboard.ui.analytics import valid_online_forecasts


def build_conformal_artifacts(store, forecasts):
    frame = valid_online_forecasts(forecasts)
    if 'model' in frame:
        frame = frame[frame.model.ne('tabpfn_online')]
    bands = conformal_intervals(frame)
    directory = store.dashboard_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'conformal_forecasts.parquet'
    temporary = path.with_suffix('.parquet.tmp')
    bands.to_parquet(temporary, index=False)
    temporary.replace(path)
    diagnostics = interval_diagnostics(bands)
    diagnostics.to_csv(directory / 'conformal_coverage.csv', index=False)
    metadata = dict(generated_at_utc=datetime.now(UTC).isoformat(), rows=len(bands),
                    calibrated_rows=int(bands.lower_90.notna().sum()), levels=list(LEVELS),
                    window_runs=WINDOW_RUNS, minimum_runs=MIN_RUNS, horizon_bucket_hours=6,
                    reporting_lag_hours=REPORTING_LAG_HOURS,
                    method='Rolling absolute-error conformal; pointwise bands',
                    historical_actuals='Current stored revisions; reporting-lag availability assumed',
                    coverage='Nominal target; exchangeability is not assumed for time-series guarantees')
    (directory / 'conformal_status.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return path, metadata
