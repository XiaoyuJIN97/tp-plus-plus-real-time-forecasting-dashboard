"""Report retained horizon/observation coverage per zone, task and model."""
import numpy as np
import pandas as pd
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp, forecast_horizon_hours


def history_coverage(forecasts):
    frame=forecasts.copy()
    keys=['zone','target','model','run_date']
    frame['run_date']=frame.run_date.astype(str)
    frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True,errors='coerce')
    days=frame.run_date.unique()
    starts={d:brussels_cutoff_timestamp(pd.Timestamp(d).date()) for d in days}
    horizons={d:forecast_horizon_hours(d) for d in days}
    offsets=(frame.timestamp-frame.run_date.map(starts))/pd.Timedelta(hours=1)
    frame['_grid']=offsets.ge(0)&offsets.lt(frame.run_date.map(horizons))&offsets.mod(1).eq(0)
    frame['_finite']=np.isfinite(pd.to_numeric(frame.forecast_mw,errors='coerce')).fillna(False)
    frame['_observed']=np.isfinite(pd.to_numeric(frame.get('actual_mw',pd.Series(np.nan,index=frame.index)),errors='coerce')).fillna(False)
    runs=frame.groupby(keys,observed=True).agg(rows=('timestamp','size'),unique=('timestamp','nunique'),
        grid=('_grid','all'),finite=('_finite','all'),observed=('_observed','all')).reset_index()
    expected=runs.run_date.map(horizons)
    runs['full']=runs.rows.eq(expected)&runs.unique.eq(expected)&runs.grid&runs.finite
    runs['fully_observed']=runs.full&runs.observed
    runs['incomplete']=~runs.full
    runs['first_full_run_date']=runs.run_date.where(runs.full)
    runs['last_full_run_date']=runs.run_date.where(runs.full)
    return runs.groupby(keys[:3],observed=True).agg(
        first_stored_run_date=('run_date','min'),last_stored_run_date=('run_date','max'),
        stored_runs=('run_date','size'),full_horizon_runs=('full','sum'),
        fully_observed_runs=('fully_observed','sum'),incomplete_horizon_runs=('incomplete','sum'),
        first_full_run_date=('first_full_run_date','min'),last_full_run_date=('last_full_run_date','max')).reset_index()
