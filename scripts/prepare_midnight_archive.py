"""Build local replay inputs from retained collector files and PSSD snapshots.

PSSD is read-only. Output is a new/reusable staging directory, never production
forecast data. Snapshot revisions are retrospective inputs, not as-issued vintages.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rt_forecast_dashboard.model_onboarding.contracts import staging_path
from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.data.entsoe_realtime_archive import ACTUAL_VARIABLES, FORECAST_VARIABLES

COUNTRIES = dict(BE='Belgium', FR='France', DE='Germany', NL='Netherlands',
                 DK1='Denmark_DK1', DK2='Denmark_DK2', ES='Spain', PT='Portugal')


def hourly(path):
    frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame['value'] = pd.to_numeric(frame.value, errors='coerce')
    return frame.set_index('timestamp_utc').value.resample('h').mean().dropna()


def prepare(collector, pssd, output, start, end):
    collector, pssd, output = map(Path, (collector, pssd, output))
    output = staging_path(output)
    if output == collector.resolve() or output.is_relative_to(collector.resolve()) or output.is_relative_to(pssd.resolve()):
        raise ValueError('Output must be separate from collector and PSSD sources')
    first = pd.Timestamp(start, tz='UTC')-pd.Timedelta(hours=2208+72)
    last = pd.Timestamp(end, tz='UTC')+pd.Timedelta(days=2)
    audit = []
    for zone, spec in zones().items():
        for target in spec['targets']:
            for mapping in (ACTUAL_VARIABLES, FORECAST_VARIABLES):
                variable = mapping[target]
                destination = output/'data/hourly'/zone/variable/f'{first.year}.parquet'
                pieces = []
                for year in range(first.year,last.year+1):
                    path = collector/'data/raw'/zone/variable/f'{year}.csv'
                    if path.exists():
                        pieces.append(hourly(path))
                # Country-name folders hold snapshots moved off the collector.
                candidates = sorted((pssd/'updates'/COUNTRIES[zone]/variable).glob('*/*/*.csv'))
                selected = {}
                for path in candidates:
                    stamp = pd.to_datetime(path.stem, format='%Y%m%dT%H%M%SZ', utc=True, errors='coerce')
                    if pd.isna(stamp) or not first <= stamp <= last+pd.Timedelta(days=7):
                        continue
                    # At least one snapshot per day; do not assume request windows
                    # or data retention are identical for every country/variable.
                    selected[stamp.date()] = path
                for path in selected.values():
                    pieces.append(hourly(path))
                for year in range(first.year,last.year+1):
                    path = collector/'data/hourly'/zone/variable/f'{year}.parquet'
                    if path.exists():
                        pieces.append(hourly(path))
                if not pieces:
                    audit.append(dict(zone=zone,target=target,variable=variable,hours=0,snapshots=0))
                    continue
                series = pd.concat(pieces)
                series = series[~series.index.duplicated(keep='last')].sort_index()
                series = series.loc[(series.index>=first)&(series.index<last)]
                frame = series.rename('value').rename_axis('timestamp_utc').reset_index()
                for year, part in frame.groupby(frame.timestamp_utc.dt.year):
                    destination = output/'data/hourly'/zone/variable/f'{year}.parquet'
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    part.to_parquet(destination,index=False)
                audit.append(dict(zone=zone,target=target,variable=variable,hours=len(frame),snapshots=len(selected),
                                  first=str(series.index.min()),last=str(series.index.max())))
                print(zone,variable,len(frame),'hours',len(selected),'PSSD snapshots',flush=True)
    pd.DataFrame(audit).to_csv(output/'input_audit.csv',index=False)
    return pd.DataFrame(audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('collector','pssd','output','start','end'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    prepare(**vars(args))
