"""Validate and stage a DK solar history repair without modifying source data.

Supply full-horizon native replays, the frozen repair plan, recovered archived
inputs and the explicit-zero evidence ledger. Other scopes and already-valid
model forecasts remain unchanged. Artifact rebuild/publication is a later step.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import is_complete_forecast_window,brussels_cutoff_timestamp
from rt_forecast_dashboard.ui.analytics import valid_online_forecasts
from rt_forecast_dashboard.model_onboarding.contracts import staging_path

KEYS=['run_date','zone','target','model']


def stage(source,replay,plan,inputs,output):
    source=Path(source).resolve();output=staging_path(output)
    if output.exists() or output.is_relative_to(source):raise ValueError('Use a new isolated staging directory')
    expected=pd.read_csv(plan,dtype={'run_date':str})[KEYS]
    if expected.duplicated().any() or not expected.zone.isin(['DK1','DK2']).all() or not expected.target.eq('solar').all():
        raise ValueError('Plan must contain unique DK solar groups only')
    incoming=ForecastStore(Path(replay)).read_forecasts()
    incoming.run_date=incoming.run_date.astype(str)
    if set(map(tuple,incoming[KEYS].drop_duplicates().to_numpy()))!=set(map(tuple,expected.to_numpy())):
        raise ValueError('Replay does not cover exactly the frozen repair plan')
    if len(valid_online_forecasts(incoming)) != len(incoming):
        raise ValueError('Replay contains unregistered or invalid forecast groups')
    for key,part in incoming.groupby(KEYS):
        origin=brussels_cutoff_timestamp(pd.Timestamp(key[0]).date())
        if (not is_complete_forecast_window(part.timestamp,key[0]) or not part.source.eq('historical_mock_replay').all()
                or not part.context_hours.eq(2208).all()
                or not pd.to_datetime(part.context_end,utc=True).eq(origin-pd.Timedelta(hours=1)).all()
                or not np.isfinite(part[['forecast_mw','actual_mw']].to_numpy(float)).all()
                or part.forecast_mw.lt(0).any()):
            raise ValueError(f'Incomplete or invalid historical replay: {key}')
    output.mkdir(parents=True)
    for folder in ['forecasts','dashboard']:
        if (source/folder).exists():shutil.copytree(source/folder,output/folder)
    store=ForecastStore(output)
    before=store.read_forecasts();valid=valid_online_forecasts(before)
    existing=valid[KEYS].drop_duplicates().assign(_preserve=True)
    imported=incoming.merge(existing,on=KEYS,how='left')
    imported=imported[imported._preserve.isna()].drop(columns='_preserve')
    for day,part in imported.groupby('run_date'):store.append_forecasts(part,str(day))
    # Apply one frozen actual series consistently to every DK solar model in
    # the repaired origins. Original known observations are retained in inputs;
    # the separate recovery helper can add only published zero evidence.
    canonical=[]
    for z in ['DK1','DK2']:
        f=pd.read_parquet(Path(inputs)/'entsoe/data/hourly'/z/'actual_solar_generation/2026.parquet')
        f=f.rename(columns={'timestamp_utc':'timestamp','value':'canonical_actual_mw'})
        f['timestamp']=pd.to_datetime(f.timestamp,utc=True);f['zone']=z;canonical.append(f)
    lookup=pd.concat(canonical,ignore_index=True)
    recovered_path=Path(inputs).parent/'recovered_published_zero_actuals.parquet'
    zeros=pd.read_parquet(recovered_path)
    evidence=set(zip(zeros.zone,pd.to_datetime(zeros.timestamp_utc,utc=True)))
    for day in sorted(expected.run_date.unique()):
        path=store.forecast_path(day);frame=pd.read_csv(path)
        scope=frame.zone.isin(['DK1','DK2'])&frame.target.eq('solar')
        part=frame[scope].copy();part['timestamp']=pd.to_datetime(part.timestamp,utc=True)
        part=part.merge(lookup,on=['zone','timestamp'],how='left',validate='many_to_one')
        if part.canonical_actual_mw.isna().any():raise ValueError('Canonical repair truth is incomplete')
        part['actual_mw']=part.pop('canonical_actual_mw')
        part['actual_source']=['energinet_5min_realtime_explicit_zero' if (z,t) in evidence else 'retrospective_entsoe_archive'
                               for z,t in zip(part.zone,part.timestamp)]
        part['history_repair']='dk_solar_midnight_2026-11-03'
        # Group replacement also preserves non-DK/other-target rows.
        store.append_forecasts(part,str(day))
    report=dict(planned_groups=len(expected),imported_groups=len(imported[KEYS].drop_duplicates()),
                preserved_complete_groups=len(incoming[KEYS].drop_duplicates())-len(imported[KEYS].drop_duplicates()),
                input_vintage='Retrospective archived revisions; explicit Energinet zero-hour evidence',
                first_run_date=expected.run_date.min(),last_run_date=expected.run_date.max())
    (output/'history_repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','replay','plan','inputs','output']:p.add_argument('--'+key,required=True)
    stage(**vars(p.parse_args()))
