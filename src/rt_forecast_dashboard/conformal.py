"""Rolling pointwise conformal bands for stored day-ahead point forecasts.

Calibration is per model/zone/target and six-hour horizon bucket. A run only
enters calibration after all delivery outcomes have finished, plus a reporting lag.
Historical actual publication/revision vintages are not available in this store;
this is retrospective calibration under the explicitly stated lag assumption.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp, forecast_horizon_hours

LEVELS = (0.80, 0.90, 0.95)
WINDOW_RUNS = 60
MIN_RUNS = 20
REPORTING_LAG_HOURS = 2
KEYS = ['zone', 'target', 'model', 'run_date', 'timestamp']


def conformal_radius(scores, coverage):
    """Finite-sample order statistic, including the n+1 correction."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not 0 < coverage < 1:
        raise ValueError('Coverage must be between zero and one')
    rank = math.ceil((len(scores) + 1) * coverage)
    if not len(scores) or rank > len(scores):
        return np.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def conformal_intervals(frame, *, levels=LEVELS, window_runs=WINDOW_RUNS,
                        min_runs=MIN_RUNS, reporting_lag_hours=REPORTING_LAG_HOURS):
    if min_runs < 1 or window_runs < min_runs or reporting_lag_hours < 0:
        raise ValueError('Invalid calibration window or reporting lag')
    if not levels or any(not 0 < level < 1 for level in levels):
        raise ValueError('Invalid coverage levels')
    columns = KEYS + ['model_label', 'forecast_mw', 'actual_mw']
    out = frame.reindex(columns=columns).copy().reset_index(drop=True)
    out['timestamp'] = pd.to_datetime(out.timestamp, utc=True)
    out['run_date'] = out.run_date.astype(str)
    if out.duplicated(KEYS).any():
        raise ValueError('Duplicate forecast keys would duplicate calibration scores')
    for level in levels:
        out[f'lower_{round(level*100)}'] = np.nan
        out[f'upper_{round(level*100)}'] = np.nan
    out['calibration_runs'] = 0
    out['calibration_last_delivery'] = pd.Series(pd.NaT, index=out.index, dtype='datetime64[ns, UTC]')
    if out.empty:
        return out
    lows = {level: np.full(len(out), np.nan) for level in levels}
    highs = {level: np.full(len(out), np.nan) for level in levels}
    counts = np.zeros(len(out), dtype=np.int32)
    last_delivery = np.full(len(out), np.datetime64('NaT'), dtype='datetime64[ns]')
    origins = {day: brussels_cutoff_timestamp(pd.Timestamp(day).date()) for day in out.run_date.unique()}
    out['horizon'] = (out.timestamp - out.run_date.map(origins)) / pd.Timedelta(hours=1)
    lag = pd.Timedelta(hours=1 + reporting_lag_hours)
    for _, series in out.groupby(['zone', 'target', 'model'], observed=True, sort=False):
        runs = []
        for day, run in series.groupby('run_date', observed=True, sort=True):
            run = run.sort_values('horizon')
            horizon = forecast_horizon_hours(day)
            if len(run) != horizon or not np.array_equal(run.horizon.to_numpy(), np.arange(horizon)):
                continue
            prediction = pd.to_numeric(run.forecast_mw, errors='coerce').to_numpy(float)
            actual = pd.to_numeric(run.actual_mw, errors='coerce').to_numpy(float)
            if not np.isfinite(prediction).all():
                continue
            runs.append(dict(origin=origins[day], index=run.index.to_numpy(),
                             prediction=prediction, error=np.abs(actual-prediction),
                             observed=bool(np.isfinite(actual).all()),
                             last=run.timestamp.max(), ready=run.timestamp.max()+lag))
        for run in runs:
            eligible = [past for past in runs if past['observed']
                        and past['origin'] < run['origin'] and past['ready'] <= run['origin']][-window_runs:]
            count = len(eligible)
            counts[run['index']] = count
            if count < min_runs:
                continue
            # Runs can have 23/24/25 hours; pool errors within the same bucket.
            # Hour 25 shares the final bucket, rather than needing 20 autumns.
            last_delivery[run['index']] = max(past['last'] for past in eligible).tz_localize(None).to_datetime64()
            for first in (0, 6, 12, 18):
                end = len(run['index']) if first == 18 else first+6
                errors = np.concatenate([past['error'][first:] if first == 18
                                         else past['error'][first:first+6] for past in eligible])
                indices = run['index'][first:end]
                point = run['prediction'][first:end]
                for level in levels:
                    radius = conformal_radius(errors, level)
                    if not np.isfinite(radius):
                        continue
                    lows[level][indices] = np.maximum(0, point-radius)
                    highs[level][indices] = point+radius
    for level in levels:
        out[f'lower_{round(level*100)}'] = lows[level]
        out[f'upper_{round(level*100)}'] = highs[level]
    out['calibration_runs'] = counts
    out['calibration_last_delivery'] = pd.to_datetime(last_delivery, utc=True)
    return out


def interval_leaderboard(frame, level=.80):
    """Compare models on identical complete, observed and calibrated UTC runs.

    Coverage is an hourly fraction, not a whole-trajectory success rate. Width
    is averaged over the same hours. Do not rank coverage as 'higher is better'.
    """
    columns = ['zone','target','model','model_label','coverage_pct','mean_interval_width_mw','scored_runs','scored_hours']
    if frame.empty:
        return pd.DataFrame(columns=columns)
    low,high = f'lower_{round(level*100)}',f'upper_{round(level*100)}'
    data = frame.copy()
    data['timestamp'] = pd.to_datetime(data.timestamp,utc=True)
    data['run_date'] = data.run_date.astype(str)
    if data.duplicated(['zone','target','model','run_date','timestamp']).any():
        raise ValueError('Duplicate interval forecast keys')
    for col in ('actual_mw',low,high):
        data[col] = pd.to_numeric(data[col],errors='coerce')
    rows = []
    for (zone,target),series in data.groupby(['zone','target'],observed=True):
        models = set(series.model.dropna().astype(str))
        complete = []
        for (_,day),part in series.groupby(['model','run_date'],observed=True):
            times = part.timestamp.sort_values()
            expected = forecast_horizon_hours(day)
            origin = brussels_cutoff_timestamp(pd.Timestamp(day).date())
            if (len(part)==expected and times.iloc[0]==origin
                    and times.diff().dropna().eq(pd.Timedelta(hours=1)).all()
                    and np.isfinite(part[['actual_mw',low,high]].to_numpy(float)).all()
                    and part[low].ge(0).all() and part[high].ge(part[low]).all()):
                complete.append(part)
        if not complete:
            continue
        matched = pd.concat(complete,ignore_index=True)
        shared = matched.groupby('run_date',observed=True).model.nunique()
        matched = matched[matched.run_date.isin(shared[shared.eq(len(models))].index)]
        if matched.empty:
            continue
        # Reject a whole run if reference truth differs across models.
        truth = matched.groupby(['run_date','timestamp'],observed=True).actual_mw.agg(['min','max'])
        bad = truth.loc[~np.isclose(truth['min'],truth['max'],rtol=1e-6,atol=1e-6)].index.get_level_values('run_date')
        matched = matched[~matched.run_date.isin(bad)]
        for model,part in matched.groupby('model',observed=True):
            rows.append(dict(zone=str(zone),target=str(target),model=str(model),
                model_label=str(part.model_label.iloc[0]),
                coverage_pct=float(100*part.actual_mw.between(part[low],part[high]).mean()),
                mean_interval_width_mw=float((part[high]-part[low]).mean()),
                scored_runs=int(part.run_date.nunique()),scored_hours=len(part)))
    return pd.DataFrame(rows,columns=columns)


def interval_diagnostics(frame, levels=LEVELS):
    """Realized pointwise coverage on the plotted evaluation window only."""
    rows = []
    for (zone, target, model, label), part in frame.groupby(['zone', 'target', 'model', 'model_label'], observed=True, sort=False):
        actual = pd.to_numeric(part.actual_mw, errors='coerce')
        for level in levels:
            low, high = part[f'lower_{round(level*100)}'], part[f'upper_{round(level*100)}']
            valid = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
            n = int(valid.sum())
            rows.append({'Zone': str(zone), 'Target': str(target), 'Model': str(label), 'Target coverage (%)': round(level*100),
                         'Observed coverage (%)': float(100*actual[valid].between(low[valid], high[valid]).mean()) if n else np.nan,
                         'Mean band width (MW)': float((high[valid]-low[valid]).mean()) if n else np.nan,
                         'Scored hours': n})
    return pd.DataFrame(rows)
