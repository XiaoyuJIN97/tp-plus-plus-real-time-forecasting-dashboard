"""Paired offline DE-solar weather-point ablation using unchanged online adapters.

No production inputs/configuration are modified. Recovered weather is a stitched
historical forecast series, not a certified issue-time forecast vintage.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from pathlib import Path
from datetime import date,timedelta,datetime,UTC
import numpy as np
import pandas as pd
import requests
from rt_forecast_dashboard.config import model_registry,model_supports_scope,features
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp,forecast_horizon_hours
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.models.adapters import make_adapter
from rt_forecast_dashboard.models.foundation_online import close_workers
from scripts.batch_replay_midnight import prepare_scope

SESSION=requests.Session()
MODELS=['tso_reference','persistence','chronos2_online','timesfm3_online','xgboost_online','t0_gamma_online','timestral_online']
ENDPOINT='https://historical-forecast-api.open-meteo.com/v1/forecast'
VARIABLES=['temperature_2m','shortwave_radiation']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_grid(day):
    return pd.date_range(brussels_cutoff_timestamp(pd.Timestamp(day).date()),periods=forecast_horizon_hours(day),freq='h')


def weather_request(points,start,end):
    params=dict(latitude=','.join(str(float(v)) for v in points.gfs_lat),longitude=','.join(str(float(v)) for v in points.gfs_lon),
                hourly=','.join(VARIABLES),start_date=str(start),end_date=str(end),timezone='UTC')
    last=None
    for attempt in range(2):
        try:
            time.sleep(.5);r=SESSION.get(ENDPOINT,params=params,timeout=25);r.raise_for_status();payload=r.json()
            if isinstance(payload,dict):payload=[payload]
            if len(payload)!=len(points):raise ValueError('Weather location count mismatch')
            result=None
            for point,p in zip(points.itertuples(index=False),payload,strict=True):
                h=p['hourly'];d=pd.DataFrame({'timestamp_utc':pd.to_datetime(h['time'],utc=True)})
                for v in VARIABLES:
                    if len(h[v])!=len(d):raise ValueError('Weather variable length mismatch')
                    d[f'{v}_p{int(point.point)}']=pd.to_numeric(h[v],errors='coerce')
                result=d if result is None else result.merge(d,on='timestamp_utc',validate='one_to_one')
            grid=pd.date_range(start,end+timedelta(days=1),freq='h',tz='UTC',inclusive='left')
            if not pd.DatetimeIndex(result.timestamp_utc).equals(grid) or not np.isfinite(result.drop(columns='timestamp_utc')).all().all():
                raise ValueError('Incomplete/nonfinite weather grid')
            return result
        except Exception as e:last=e
    if start<end:
        mid=start+timedelta(days=(end-start).days//2)
        return pd.concat([weather_request(points,start,mid),weather_request(points,mid+timedelta(days=1),end)],ignore_index=True)
    raise RuntimeError(f'Weather request failed for {start}: {last}')


def fetch_weather(root,arm,points,start,end):
    points=points.sort_values('point')
    if len(points)!=4 or set(points.point)!={1,2,3,4}:raise ValueError('The unchanged online configuration requires point IDs 1–4')
    root.mkdir(parents=True,exist_ok=True);points.to_csv(root/f'{arm}_points.csv',index=False)
    chunks=root/'weather_chunks'/arm;chunks.mkdir(parents=True,exist_ok=True);paths=[];cursor=start
    while cursor<=end:
        stop=min(cursor+timedelta(days=6),end);p=chunks/f'{cursor}_{stop}.parquet'
        if not p.exists():
            d=weather_request(points,cursor,stop);d.to_parquet(p,index=False)
            p.with_suffix('.json').write_text(json.dumps(dict(endpoint=ENDPOINT,model_policy='automatic_best_match',
                fetched_at_utc=datetime.now(UTC).isoformat(),vintage='retrospective_stitched_historical_forecast',
                points=points[['point','gfs_lat','gfs_lon']].to_dict('records'),sha256=sha(p)),indent=2))
        paths.append(p);cursor=stop+timedelta(days=1)
        print('weather',arm,stop,flush=True)
    d=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True).sort_values('timestamp_utc')
    d['deg_proxy']=(d.filter(regex='^temperature').mean(axis=1)-18).abs()
    out=root/'inputs'/arm/'weather/data/raw/DE/solar/2026.csv';out.parent.mkdir(parents=True,exist_ok=True);d.to_csv(out,index=False)


def prepare_series(root,saved,online,start,end):
    root.mkdir(parents=True,exist_ok=True);downloads=root/'downloads';downloads.mkdir(exist_ok=True)
    # Freeze one common revised actual/TSO history for every weather arm.
    for variable in ['actual_solar_generation','forecast_solar_generation']:
        p=downloads/f'{variable}.parquet'
        if not p.exists():
            u=f'https://raw.githubusercontent.com/Energy-Data-Science/entsoe-realtime-data/data/data/hourly/DE/{variable}/2026.parquet'
            r=SESSION.get(u,timeout=45);r.raise_for_status();p.write_bytes(r.content)
        old=pd.read_parquet(saved/'entsoe/data/hourly/DE'/variable/'2026.parquet');new=pd.read_parquet(p)
        d=pd.concat([old,new],ignore_index=True);d['timestamp_utc']=pd.to_datetime(d.timestamp_utc,utc=True)
        d=d.drop_duplicates('timestamp_utc',keep='last').sort_values('timestamp_utc')
        for arm in ['baseline','candidate']:
            out=root/'inputs'/arm/'entsoe/data/hourly/DE'/variable/'2026.parquet';out.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(out,index=False)
        if variable.startswith('actual'):
            truth=d[['timestamp_utc','value']].rename(columns={'value':'actual_mw'}).assign(zone='DE',target='solar')
            truth.to_parquet(root/'actuals.parquet',index=False)
    rows=[];missing=[]
    for day in pd.date_range(start,end):
        label=str(day.date());p=online/'forecasts'/f'forecasts_{label}.csv';grid=expected_grid(label)
        if not p.exists():missing.append(dict(run_date=label,reason='No saved online forecast file'));continue
        d=pd.read_csv(p,low_memory=False);d=d[(d.zone=='DE')&(d.target=='solar')&(d.model=='tso_reference')].copy()
        d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
        d=d.drop_duplicates('timestamp',keep='last').set_index('timestamp').reindex(grid)
        if d.forecast_mw.isna().any():missing.append(dict(run_date=label,reason='Incomplete saved online TSO curve'));continue
        rows.append(pd.DataFrame({'run_date':label,'timestamp':grid,'tso_forecast_mw':d.forecast_mw.to_numpy()}))
    pd.concat(rows,ignore_index=True).to_parquet(root/'online_tso_by_origin.parquet',index=False)
    pd.DataFrame(missing,columns=['run_date','reason']).to_csv(root/'missing_online_tso.csv',index=False)
    fingerprint={str(p):sha(p) for p in [Path('config/features.yml'),Path('config/model_registry.yml'),Path('config/ensemble_registry.yml'),Path('src/rt_forecast_dashboard/models/foundation_runtime.py')]}
    (root/'configuration_hashes.json').write_text(json.dumps(fingerprint,indent=2))
    print('frozen TSO origins',len(rows),'missing',len(missing),flush=True)


def prepare_arms(root,start,end):
    curves=pd.read_parquet(root/'online_tso_by_origin.parquet');curves.timestamp=pd.to_datetime(curves.timestamp,utc=True)
    days=[str(d.date()) for d in pd.date_range(start,end)];prepared={};issues=[]
    for arm in ['baseline','candidate']:
        scopes,errors=prepare_scope(root/'inputs'/arm,'DE','solar',days)
        issues.extend(dict(arm=arm,**e) for e in errors)
        for day in list(scopes):
            context,future,quality,truth=scopes[day];saved=curves[curves.run_date.eq(day)].set_index('timestamp').tso_forecast_mw.reindex(pd.DatetimeIndex(future.timestamp))
            if saved.isna().any() or not np.isfinite(truth).all():
                issues.append(dict(arm=arm,run_date=day,stage='score_eligibility',error='Incomplete saved online TSO or observed actual delivery'))
                del scopes[day];continue
            future['tso_forecast_mw']=saved.to_numpy()
            assert context.timestamp.max()<future.timestamp.min() and 'actual_mw' not in future
        prepared[arm]=scopes
    common=sorted(set(prepared['baseline'])&set(prepared['candidate']))
    pd.DataFrame(issues).to_csv(root/'preparation_issues.csv',index=False)
    quality=[]
    for arm in prepared:
        for day in common:
            c,f,q,t=prepared[arm][day];quality.append(dict(arm=arm,run_date=day,context_missing=q.missing_hours,context_imputed=q.imputed_hours,methods=q.methods,hours=len(f)))
    pd.DataFrame(quality).to_csv(root/'context_quality.csv',index=False)
    (root/'eligible_origins.json').write_text(json.dumps(common,indent=2));return prepared,common


def validate_configuration(root):
    for name,digest in json.loads((root/'configuration_hashes.json').read_text()).items():
        if sha(name)!=digest:
            raise RuntimeError(f'Online configuration differs from frozen study: {name}')


def run(root,start,end,models=None):
    validate_configuration(root)
    import torch
    torch.set_num_threads(2)
    prepared,days=prepare_arms(root,start,end);registry=model_registry()
    timing=pd.read_csv(root/'inference_timing.csv').to_dict('records') if (root/'inference_timing.csv').exists() else []
    failures=[]
    print('eligible paired origins',len(days),flush=True)
    for key in (models or MODELS):
        config=registry[key];adapter=make_adapter(config);case,covs=covariates_for('solar','DE',key)
        for day in days:
            for arm in ['baseline','candidate']:
                store=ForecastStore(root/'predictions'/arm);path=store.forecast_path(day)
                if path.exists():
                    old=pd.read_csv(path)
                    if len(old[old.model.eq(key)])==forecast_horizon_hours(day):continue
                context,future,quality,truth=prepared[arm][day]
                before=time.perf_counter()
                try:
                    pred=np.asarray(adapter.predict(future.copy(),'solar',context=context.copy(),covariates=covs),dtype=float)
                    elapsed=time.perf_counter()-before
                    if pred.shape!=(len(future),) or not np.isfinite(pred).all():raise ValueError('Invalid adapter output')
                    frame=pd.DataFrame(dict(run_date=day,run_at=datetime.now(UTC).isoformat(),source='offline_de_solar_weather_ablation',
                        forecast_window='next_day_midnight_v1',weather_arm=arm,input_vintage='paired_retrospective_weather',
                        zone='DE',target='solar',model=key,model_label=config.get('label',key),covariate_case=case,context_hours=2208,
                        timestamp=future.timestamp,forecast_mw=pred,tso_forecast_mw=future.tso_forecast_mw,actual_mw=truth,
                        context_start=context.timestamp.min(),context_end=context.timestamp.max(),
                        context_missing_hours=quality.missing_hours,context_imputed_hours=quality.imputed_hours,
                        context_imputation_method=quality.methods,data_quality=quality.quality))
                    store.append_forecasts(frame,day);timing.append(dict(arm=arm,model=key,run_date=day,seconds=elapsed,hours=len(future),device='cpu'))
                except Exception as e:
                    failures.append(dict(arm=arm,model=key,run_date=day,error=str(e)));print('FAILED',failures[-1],flush=True)
                pd.DataFrame(timing).to_csv(root/'inference_timing.csv',index=False)
                pd.DataFrame(failures,columns=['arm','model','run_date','error']).to_csv(root/'inference_failures.csv',index=False)
            print('inference',key,day,flush=True)
        close_workers()
    # Run the unchanged per-scope ensemble implementation against each arm.
    from rt_forecast_dashboard.pipeline.build_selected_ensembles import build_selected_ensembles
    for arm in ['baseline','candidate']:
        build_selected_ensembles(ForecastStore(root/'predictions'/arm),actuals_path=root/'actuals.parquet',run_dates=days,source='offline_weather_ensemble')
    validate_configuration(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','weather','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--saved-inputs',type=Path);p.add_argument('--online-data',type=Path,default=Path('data'))
    p.add_argument('--start',default='2026-07-24');p.add_argument('--end',default='2026-10-31');p.add_argument('--arm',choices=['baseline','candidate']);p.add_argument('--points',type=Path);p.add_argument('--models')
    a=p.parse_args();root=a.root.resolve();start=pd.Timestamp(a.start).date();end=pd.Timestamp(a.end).date()
    if a.command=='prepare':prepare_series(root,a.saved_inputs.resolve(),a.online_data.resolve(),start,end)
    elif a.command=='weather':
        points=pd.read_csv(a.points);points=points[(points.country=='DE')&(points.type=='solar')]
        first=(brussels_cutoff_timestamp(start)-pd.Timedelta(hours=2208)).date();last=(expected_grid(end)[-1]).date()
        fetch_weather(root,a.arm,points,first,last)
    else:run(root,start,end,a.models.split(',') if a.models else None)

if __name__=='__main__':main()
