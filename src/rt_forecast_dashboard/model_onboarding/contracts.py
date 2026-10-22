from __future__ import annotations

from rt_forecast_dashboard.time_utils import forecast_horizon_hours

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


KEYS = ['zone', 'target', 'run_date', 'timestamp']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frames(context, future, covariates, *, context_hours=2208, horizon=None):
    """Return independent, strictly selected frames; future truth is never forwarded."""
    if len(set(covariates)) != len(covariates) or any(c in {'timestamp', 'actual_mw'} for c in covariates):
        raise ValueError('Invalid covariate names')
    past = pd.DataFrame(context).loc[:, ['timestamp', 'actual_mw', *covariates]].copy()
    next_ = pd.DataFrame(future).loc[:, ['timestamp', *covariates]].copy()
    if horizon is None:
        origin = pd.to_datetime(next_.timestamp, utc=True).iloc[0]
        horizon = forecast_horizon_hours(origin.tz_convert('Europe/Brussels').date())
    for frame, length in ((past, context_hours), (next_, horizon)):
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise')
        if len(frame) != length or frame.timestamp.isna().any() or frame.timestamp.duplicated().any():
            raise ValueError(f'Expected exactly {length} unique timestamps')
        if not frame.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError('Timestamps must be sorted and hourly')
        numeric = frame.drop(columns='timestamp').apply(pd.to_numeric, errors='raise')
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError('Inputs must be finite; imputation belongs in audited preparation')
        frame[numeric.columns] = numeric
    if past.timestamp.max()+pd.Timedelta(hours=1) != next_.timestamp.min():
        raise ValueError('Context must end exactly one hour before forecast origin')
    return past, next_


def prediction(values, horizon):
    result = np.asarray(values, dtype=float)
    if result.shape != (horizon,) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f'Expected {horizon} finite, non-negative MW predictions')
    return result


def records(frame):
    frame = frame.copy()
    frame['timestamp'] = frame.timestamp.map(lambda value: value.isoformat())
    return frame.to_dict('records')


def inference_digest(manifest_path):
    """Keep approval edits separate from the immutable evaluated model identity."""
    import json
    import yaml
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    identity = {k:v for k,v in manifest.items() if k not in {'review','operations','licenses'}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def staging_path(path):
    """Only the explicit publisher may write beneath the production data root."""
    import os
    from .process import ROOT
    result=Path(path).resolve()
    data=Path(os.getenv('FORECAST_DATA_DIR',str(ROOT/'data')))
    data=(data if data.is_absolute() else ROOT/data).resolve()
    if result==data or result.is_relative_to(data):
        raise ValueError('Staging/evaluation commands cannot write production data')
    return result
