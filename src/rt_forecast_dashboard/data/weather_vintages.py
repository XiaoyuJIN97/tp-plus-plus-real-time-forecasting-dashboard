"""Weather available at a historical cutoff, from original collection snapshots.

A successful collection proves availability by that time. A model initialization
or delivery timestamp does not. Annual merged files are deliberately never read.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .weather_archive import TARGET_MAP


class WeatherVintageError(ValueError):
    pass


class WeatherVintages:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest = self.root / 'data/update_manifest.csv'
        if not manifest.is_file():
            raise WeatherVintageError('No collection manifest; merged annual files are not replay vintages')
        self.manifest = pd.read_csv(manifest)
        required = {'country','target','path','collection_time_utc'}
        if not required.issubset(self.manifest):
            raise WeatherVintageError('Collection manifest lacks availability evidence')
        self.manifest['available_at'] = pd.to_datetime(self.manifest.collection_time_utc, utc=True, errors='coerce')
        self.manifest = self.manifest.dropna(subset=['available_at','path'])

    @lru_cache(maxsize=512)
    def _read(self, relative: str):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root / 'data/updates') or not path.is_file():
            raise WeatherVintageError('Missing original snapshot or invalid snapshot path: '+relative)
        payload = path.read_bytes()
        frame = pd.read_csv(path)
        if not {'timestamp_utc','updated_at_utc'}.issubset(frame):
            raise WeatherVintageError('Snapshot lacks per-row collection evidence: '+relative)
        frame['timestamp'] = pd.to_datetime(frame.timestamp_utc, utc=True, errors='coerce')
        frame['_collected'] = pd.to_datetime(frame.updated_at_utc, utc=True, errors='coerce')
        if frame.timestamp.isna().any() or frame._collected.isna().any() or frame.timestamp.duplicated().any():
            raise WeatherVintageError('Invalid/duplicate snapshot timestamps: '+relative)
        columns = [c for c in frame.select_dtypes(include=np.number) if c not in ['country','target']]
        return frame.set_index('timestamp')[columns+['_collected']], hashlib.sha256(payload).hexdigest()

    def select(self, zone: str, target: str, cutoff, context_hours: int, horizon: int):
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None:
            raise WeatherVintageError('Cutoff must be timezone-aware')
        cutoff = cutoff.tz_convert('UTC')
        future_grid = pd.date_range(cutoff, periods=horizon, freq='h', name='timestamp')
        past_grid = pd.date_range(cutoff-pd.Timedelta(hours=context_hours), periods=context_hours, freq='h', name='timestamp')
        rows = self.manifest[(self.manifest.country.eq(zone)) & self.manifest.target.eq(TARGET_MAP[target])
                             & self.manifest.available_at.le(cutoff)].sort_values(['available_at','path'])
        if rows.empty:
            raise WeatherVintageError('No snapshot collected before cutoff')
        future = None; past = None; evidence = []; errors = []
        # Newest first. Future hours must all come from one coherent snapshot.
        for row in rows.iloc[::-1].itertuples():
            try:
                frame, digest = self._read(row.path)
            except WeatherVintageError as exc:
                errors.append(str(exc)); continue
            eligible = frame[frame._collected.le(cutoff)].drop(columns='_collected')
            if eligible.empty: continue
            used = False
            future_used = False
            phase = str(getattr(row,'phase',''))
            vintage = str(getattr(row,'vintage',''))
            endpoint = str(getattr(row,'endpoint',''))
            historical = phase == 'history' or 'retrospective' in vintage or 'historical-forecast' in endpoint
            candidate = eligible.reindex(future_grid)
            if future is None and not historical and len(candidate.columns) and np.isfinite(candidate.to_numpy(float)).all():
                future = candidate; used = True; future_used = True
            candidate = eligible.reindex(past_grid)
            if past is None:
                past = candidate
                used |= candidate.notna().any().any()
            else:
                # Preserve latest collected complete row, never combine variables
                # from conflicting snapshots at one delivery timestamp.
                missing = ~np.isfinite(past.to_numpy(float)).all(axis=1)
                complete = np.isfinite(candidate.reindex(columns=past.columns).to_numpy(float)).all(axis=1)
                take = missing & complete
                if take.any():
                    past.loc[take] = candidate.reindex(columns=past.columns).loc[take]
                    used = True
            if used:
                evidence.append(dict(path=row.path,sha256=digest,available_at=row.available_at.isoformat(),role='future_and_context' if future_used else 'context'))
            if future is not None and past is not None and np.isfinite(past.to_numpy(float)).all():
                break
        if future is None:
            raise WeatherVintageError('No pre-cutoff snapshot covers the full future horizon'+ ('; '+errors[0] if errors else ''))
        missing = context_hours if past is None else int((~np.isfinite(past.to_numpy(float)).all(axis=1)).sum())
        if missing:
            raise WeatherVintageError(f'Pre-cutoff snapshots lack {missing}/{context_hours} context weather hours')
        provenance = dict(policy='weather_collection_asof_v1',cutoff_utc=cutoff.isoformat(),snapshots=evidence,
                          other_inputs='TSO and actual revision vintages require separate certification')
        for frame in (past,future): frame.attrs['weather_vintage'] = provenance
        return past, future, provenance

    def joined(self, zone, target, cutoff, context_hours, horizon):
        past,future,provenance = self.select(zone,target,cutoff,context_hours,horizon)
        result = pd.concat([past,future]);result.attrs['weather_vintage'] = provenance
        return result
