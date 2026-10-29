"""Preflight historical windows without model inference or upstream requests."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp, forecast_horizon_hours
from rt_forecast_dashboard.context_quality import repair_context
from rt_forecast_dashboard.data.entsoe_realtime_archive import ACTUAL_VARIABLES,FORECAST_VARIABLES
from rt_forecast_dashboard.data.weather_archive import TARGET_MAP


def audit(source, inputs, output, start, end):
    source, inputs, output = map(Path,(source,inputs,output))
    saved=ForecastStore(source).read_forecasts()
    compact=source/'dashboard/forecasts_recent.parquet'
    if compact.exists():saved=pd.concat([saved,pd.read_parquet(compact)],ignore_index=True)
    saved.run_date=saved.run_date.astype(str)
    plan=saved[saved.run_date.between(start,end)][['zone','target','run_date']].drop_duplicates()
    enabled={(z,t) for z,c in zones().items() for t in c['targets']}
    plan=plan[pd.MultiIndex.from_frame(plan[['zone','target']]).isin(enabled)]
    rows=[]
    for (zone,target),scope in plan.groupby(['zone','target']):
        series={}
        for field,mapping in [('actual_mw',ACTUAL_VARIABLES),('tso_forecast_mw',FORECAST_VARIABLES)]:
            parts=[]
            for p in sorted((inputs/'entsoe/data/hourly'/zone/mapping[target]).glob('*.parquet')):
                f=pd.read_parquet(p);f['timestamp']=pd.to_datetime(f.timestamp_utc,utc=True)
                parts.append(f.set_index('timestamp').value)
            s=pd.concat(parts) if parts else pd.Series(dtype=float)
            series[field]=s[~s.index.duplicated(keep='last')].sort_index()
        weather_files=sorted((inputs/'weather/data/raw'/zone/TARGET_MAP[target]).glob('*.csv'))
        w=pd.concat([pd.read_csv(p) for p in weather_files],ignore_index=True) if weather_files else pd.DataFrame(columns=['timestamp_utc'])
        w['timestamp']=pd.to_datetime(w.timestamp_utc,utc=True);w=w.drop_duplicates('timestamp',keep='last').set_index('timestamp')
        for day in sorted(scope.run_date):
            origin=brussels_cutoff_timestamp(pd.Timestamp(day).date());h=forecast_horizon_hours(day)
            future=pd.date_range(origin,periods=h,freq='h')
            context=pd.date_range(origin-pd.Timedelta(hours=2208),periods=2208,freq='h')
            reasons=[]
            tso=series['tso_forecast_mw'].reindex(future)
            if tso.isna().any():reasons.append(f'missing_tso_hours={int(tso.isna().sum())}')
            if not future.isin(w.index).all():reasons.append('missing_future_weather_hours')
            if not context.isin(w.index).all():reasons.append('missing_context_weather_hours')
            try:
                frame=pd.DataFrame({k:s.reindex(context) for k,s in series.items()}).rename_axis('timestamp').reset_index()
                repair_context(frame,expected_end=context[-1],context_hours=2208)
            except Exception as exc:reasons.append(str(exc))
            rows.append(dict(zone=zone,target=target,run_date=day,hours=h,ready=not reasons,reason=';'.join(reasons)))
    result=pd.DataFrame(rows);output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    print(result.groupby(['zone','target']).ready.agg(['size','sum']).to_string())
    print('ready',int(result.ready.sum()),'of',len(result),'windows')
    if (~result.ready).any():print(result[~result.ready].head(20).to_string(index=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','inputs','output','start','end']:p.add_argument('--'+name,required=True)
    audit(**vars(p.parse_args()))
