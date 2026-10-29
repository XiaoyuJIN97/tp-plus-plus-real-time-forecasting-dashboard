"""Resumable full-horizon rerun into staging, with explicit retrospective provenance.

Every candidate is regenerated from its original context endpoint. Existing
24-hour forecasts are never padded, or copied from a later forecast origin.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import pandas as pd
from rt_forecast_dashboard.pipeline.run_daily import run_daily_forecast
from rt_forecast_dashboard.pipeline.check_forecast_complete import complete_forecast_groups
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.model_onboarding.contracts import staging_path
from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp, forecast_horizon_hours


def _replay(source, output, start, end, models=None, selected_scopes=None, plan_path=None, stage="replay"):
    source, output = Path(source).resolve(), staging_path(output)
    if source == output or output.is_relative_to(source):
        raise ValueError('Replay requires a separate staging directory')
    if plan_path:
        original = pd.read_csv(plan_path, dtype={'run_date':str})
    else:
        original = ForecastStore(source).read_forecasts()
        compact = source/'dashboard/forecasts_recent.parquet'
        if compact.exists():
            original = pd.concat([original,pd.read_parquet(compact)],ignore_index=True)
    original.run_date = original.run_date.astype(str)
    plan = original[original.run_date.between(start,end)]
    plan = plan[~plan.model.isin(['ensemble_selected','tabpfn_online'])]
    enabled={(z,t) for z,c in zones().items() for t in c['targets']}
    plan=plan[pd.MultiIndex.from_frame(plan[['zone','target']]).isin(enabled)]
    if models:
        plan = plan[plan.model.isin(models)]
    if selected_scopes:
        plan = plan[pd.MultiIndex.from_frame(plan[['zone','target']]).isin(selected_scopes)]
    groups = plan[['run_date','zone','target','model']].drop_duplicates()
    if groups.empty:
        raise ValueError('No historical model groups match the requested replay')
    os.environ['FORECAST_DATA_DIR'] = str(output)
    os.environ['FORECAST_ARCHIVE_ONLY'] = 'true'
    store = ForecastStore(output)
    audit = []
    for day, scope_plan in groups.groupby('run_date',sort=True):
        day_start = time.monotonic()
        for (zone,target), model_plan in scope_plan.groupby(['zone','target'],sort=True):
            expected_models = set(model_plan.model)
            result = run_daily_forecast(pd.Timestamp(day).date(),model_keys=expected_models,
                         clear_issues=False,only_missing=True,scopes={(zone,target)})
            if not result.empty:
                result['source'] = 'offline_midnight_backtest' if stage == 'offline' else 'historical_mock_replay'
                result['forecast_window'] = 'next_day_midnight_v1'
                result['input_vintage'] = 'retrospective_archived_revisions'
                # Truth is attached only after inference; it never enters future features.
                from rt_forecast_dashboard.data.entsoe_realtime_archive import EntsoeRealtimeArchive
                origin = brussels_cutoff_timestamp(pd.Timestamp(day).date())
                truth = EntsoeRealtimeArchive().fetch_actuals(zone=zone,target=target,start=origin,
                         end=origin+pd.Timedelta(hours=forecast_horizon_hours(day)))
                result = result.drop(columns='actual_mw').merge(truth,on='timestamp',how='left')
                store.append_forecasts(result,day)
            complete = complete_forecast_groups(store.forecast_path(day))
            for model in expected_models:
                audit.append(dict(run_date=day,zone=zone,target=target,model=model,
                                  complete=(zone,target,model) in complete))
            report = pd.DataFrame(audit)
            report.to_csv(output/'midnight_replay_audit.csv',index=False)
            print(day,zone,target,'complete',sum((zone,target,m) in complete for m in expected_models),
                  '/',len(expected_models),'seconds',round(time.monotonic()-day_start,1),flush=True)
    report = pd.DataFrame(audit)
    (output/'midnight_replay_status.json').write_text(json.dumps(dict(start=start,end=end,
        groups=len(report),complete_groups=int(report.complete.sum()),
        publishable=bool(report.complete.all()),window='next_day_midnight_v1'),indent=2)+'\n')
    return bool(report.complete.all())


def replay(*args, **kwargs):
    previous = {key:os.environ.get(key) for key in ('FORECAST_DATA_DIR','FORECAST_ARCHIVE_ONLY')}
    try:
        return _replay(*args, **kwargs)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','start','end'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--plan',help='Optional explicit run_date,zone,target,model plan CSV')
    parser.add_argument('--stage',choices=['replay','offline'],default='replay')
    parser.add_argument('--models',help='Comma-separated registered model keys')
    parser.add_argument('--scopes',help='Comma-separated zone:target pairs')
    args=parser.parse_args()
    ok=replay(args.source,args.output,args.start,args.end,
              set(args.models.split(',')) if args.models else None,
              {tuple(s.split(':')) for s in args.scopes.split(',')} if args.scopes else None,
              plan_path=args.plan,stage=args.stage)
    raise SystemExit(0 if ok else 1)
