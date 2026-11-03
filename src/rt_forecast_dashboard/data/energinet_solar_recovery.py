"""Audited historical recovery of explicitly published DK solar zero hours.

This is not an online fallback or a rule that nighttime means zero. Only a full
hour of twelve explicit, finite zero SolarPower records from Energinet's
ElectricityProdex5MinRealtime series can recover a missing archived value.
Nonzero readings are deliberately not substituted: data revisions/definitions
can differ from the ENTSO-E series. Existing archived observations take priority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE = 'energinet_5min_realtime_explicit_zero'


def published_zero_hours(observations: pd.DataFrame) -> pd.DataFrame:
    required = {'Minutes5UTC', 'PriceArea', 'SolarPower'}
    if required - set(observations):
        raise ValueError('Missing Energinet timestamp, price area or solar power columns')
    frame = observations[list(required)].copy()
    frame['timestamp_utc'] = pd.to_datetime(frame.Minutes5UTC, utc=True, errors='raise')
    frame['SolarPower'] = pd.to_numeric(frame.SolarPower, errors='raise')
    if not frame.PriceArea.isin(['DK1', 'DK2']).all():
        raise ValueError('Only DK1/DK2 area-specific solar evidence is accepted')
    time = frame.timestamp_utc
    if time.isna().any() or not time.eq(time.dt.floor('5min')).all():
        raise ValueError('Evidence must use the exact five-minute UTC grid')
    keys = ['PriceArea', 'timestamp_utc']
    if frame.groupby(keys).SolarPower.nunique(dropna=False).gt(1).any():
        raise ValueError('Conflicting published records for the same interval')
    frame = frame.drop_duplicates(keys)
    frame['hour'] = frame.timestamp_utc.dt.floor('h')
    counts = frame.groupby(['PriceArea', 'hour']).agg(
        points=('timestamp_utc', 'size'), finite=('SolarPower', lambda x:np.isfinite(x.to_numpy(dtype=float,na_value=np.nan)).all()),
        minimum=('SolarPower', 'min'), maximum=('SolarPower', 'max'))
    accepted = counts[counts.points.eq(12) & counts.finite & counts.minimum.eq(0) & counts.maximum.eq(0)]
    result = accepted.reset_index().rename(columns={'PriceArea':'zone', 'hour':'timestamp_utc'})
    result['target'] = 'solar'
    result['actual_mw'] = 0.
    result['actual_source'] = SOURCE
    return result[['zone', 'target', 'timestamp_utc', 'actual_mw', 'actual_source']]


def fill_missing_actuals(archived: pd.DataFrame, zeros: pd.DataFrame, zone: str):
    """Return merged archived values and a ledger of newly recovered hours."""
    if zone not in {'DK1', 'DK2'}:
        raise ValueError('Only DK solar history is supported')
    current = archived.copy()
    current['timestamp_utc'] = pd.to_datetime(current.timestamp_utc, utc=True)
    if current.timestamp_utc.duplicated().any():
        raise ValueError('Duplicate archived timestamps')
    previous = current.set_index('timestamp_utc').value.sort_index()
    evidence = zeros[zeros.zone.eq(zone)].copy()
    if not evidence.empty and (not evidence.actual_mw.eq(0).all() or not evidence.actual_source.eq(SOURCE).all()):
        raise ValueError('Use only explicitly published zero-hour evidence')
    evidence = evidence[evidence.timestamp_utc.between(previous.index.min(), previous.index.max())]
    present = previous.reindex(pd.DatetimeIndex(evidence.timestamp_utc)).to_numpy()
    recovered = evidence[pd.isna(present)].copy()
    merged = previous.combine_first(recovered.set_index('timestamp_utc').actual_mw).sort_index()
    return merged.rename('value').rename_axis('timestamp_utc').reset_index(), recovered
