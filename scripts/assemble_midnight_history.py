"""Assemble validated full-horizon replay groups and retain legacy source history.

Writes staging only. Unrecoverable old windows remain archived and are explicitly
listed; they cannot enter new-horizon plots, training, or leaderboard scores.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rt_forecast_dashboard.config import zones,model_registry,model_supports_scope
from rt_forecast_dashboard.model_onboarding.contracts import staging_path
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp,forecast_horizon_hours

KEYS=['run_date','zone','target','model']


def assemble(source,inputs,output,start,end):
    source=Path(source).resolve();output=staging_path(output)
    if output.is_relative_to(source) or output==source:raise ValueError('Use a separate staging output')
    store=ForecastStore(output)
    old=ForecastStore(source).read_forecasts();compact=source/'dashboard/forecasts_recent.parquet'
    if compact.exists():old=pd.concat([old,pd.read_parquet(compact)],ignore_index=True)
    old.run_date=old.run_date.astype(str);old.timestamp=pd.to_datetime(old.timestamp,utc=True)
    old=old.drop_duplicates(KEYS+['timestamp'],keep='last')
    frames=[]
    for directory in inputs:
        f=ForecastStore(Path(directory)).read_forecasts()
        if not f.empty:frames.append(f)
    if not frames:raise ValueError('No staged replay forecasts')
    replay=pd.concat(frames,ignore_index=True);replay.run_date=replay.run_date.astype(str)
    replay=replay[replay.run_date.between(start,end)].copy()
    replay.timestamp=pd.to_datetime(replay.timestamp,utc=True)
    replay.run_at=pd.to_datetime(replay.run_at,utc=True)
    if 'inference_device' not in replay:
        replay['inference_device']='cpu'
    replay['inference_device']=replay.inference_device.fillna('cpu')
    replay=replay.sort_values('run_at').drop_duplicates(KEYS+['timestamp'],keep='last')
    registry=model_registry();enabled=zones()
    for key,f in replay.groupby(KEYS):
        day,zone,target,model=key
        cfg=registry.get(model,{})
        if (zone not in enabled or target not in enabled[zone]['targets'] or not cfg.get('enabled')
                or cfg.get('pipeline_generated') or not model_supports_scope(cfg,zone,target)):
            raise ValueError(f'Unregistered replay scope/model: {key}')
        origin=brussels_cutoff_timestamp(pd.Timestamp(day).date())
        expected=pd.date_range(origin,periods=forecast_horizon_hours(day),freq='h')
        if (not pd.DatetimeIndex(f.timestamp.sort_values()).equals(expected)
                or not np.isfinite(pd.to_numeric(f.forecast_mw,errors='coerce').to_numpy(float)).all()
                or not f.source.eq('historical_mock_replay').all()
                or not f.forecast_window.eq('next_day_midnight_v1').all()
                or not f.context_hours.eq(2208).all()
                or not pd.to_datetime(f.context_end,utc=True).eq(origin-pd.Timedelta(hours=1)).all()):
            raise ValueError(f'Invalid replay group: {key}')
    planned=old[old.run_date.between(start,end)][KEYS].drop_duplicates()
    planned=planned[planned.apply(lambda r:r.zone in enabled and r.target in enabled[r.zone]['targets']
                    and r.model in registry and registry[r.model].get('enabled') and not registry[r.model].get('pipeline_generated')
                    and r.model!='tabpfn_online' and model_supports_scope(registry[r.model],r.zone,r.target),axis=1)]
    coverage=planned.merge(replay[KEYS].drop_duplicates().assign(migrated=True),on=KEYS,how='left')
    coverage['migrated']=coverage.migrated.eq(True)
    # Group replacement, never a mixture of old and new predictions at one origin.
    replaced=replay[KEYS].drop_duplicates().assign(_replace=True)
    retained=old.merge(replaced,on=KEYS,how='left');retained=retained[retained._replace.isna()].drop(columns='_replace')
    combined=pd.concat([retained,replay],ignore_index=True)
    for day,f in combined.groupby('run_date',sort=True):
        f.to_csv(store.forecast_path(day),index=False)
    coverage.to_csv(output/'midnight_migration_coverage.csv',index=False)
    coverage[~coverage.migrated].to_csv(output/'midnight_unrecovered_groups.csv',index=False)
    summary=dict(window='next_day_midnight_v1',start=start,end=end,replay_rows=len(replay),
                 migrated_groups=int(coverage.migrated.sum()),planned_groups=len(coverage),
                 unrecovered_groups=int((~coverage.migrated).sum()),
                 all_retained_groups_migrated=bool(coverage.migrated.all()),
                 provenance='Retrospective archived revisions; not certified as-issued forecasts')
    (output/'midnight_migration_status.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return coverage


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','output','start','end']:p.add_argument('--'+key,required=True)
    p.add_argument('--inputs',nargs='+',required=True)
    assemble(**vars(p.parse_args()))
