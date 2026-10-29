"""Batch native inference for the midnight-window history migration.

Uses the online adapters, covariate policy and context repair. Only windows of
identical elapsed length are batched. Outputs stay outside production data.
"""
from __future__ import annotations
import argparse
from datetime import datetime,UTC
from pathlib import Path
import time
import numpy as np
import pandas as pd
from rt_forecast_dashboard.config import zones,model_registry,model_supports_scope
from rt_forecast_dashboard.context_quality import repair_context
from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.data.entsoe_realtime_archive import ACTUAL_VARIABLES,FORECAST_VARIABLES
from rt_forecast_dashboard.data.weather_archive import TARGET_MAP
from rt_forecast_dashboard.features import build_feature_frame
from rt_forecast_dashboard.models.adapters import make_adapter
from rt_forecast_dashboard.models.foundation_online import close_workers
from rt_forecast_dashboard.pipeline.check_forecast_complete import complete_forecast_groups
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.model_onboarding.contracts import staging_path
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp,forecast_horizon_hours


def _prepare_scope(inputs,zone,target,days):
    inputs=Path(inputs);series={}
    for field,mapping in [('actual_mw',ACTUAL_VARIABLES),('tso_forecast_mw',FORECAST_VARIABLES)]:
        frames=[pd.read_parquet(p) for p in sorted((inputs/'entsoe/data/hourly'/zone/mapping[target]).glob('*.parquet'))]
        f=pd.concat(frames,ignore_index=True);f['timestamp']=pd.to_datetime(f.timestamp_utc,utc=True)
        series[field]=f.drop_duplicates('timestamp',keep='last').set_index('timestamp').value.sort_index()
    frames=[pd.read_csv(p) for p in sorted((inputs/'weather/data/raw'/zone/TARGET_MAP[target]).glob('*.csv'))]
    w=pd.concat(frames,ignore_index=True);w['timestamp']=pd.to_datetime(w.timestamp_utc,utc=True)
    w=w.drop_duplicates('timestamp',keep='last').set_index('timestamp').select_dtypes(include=np.number)
    prepared={};issues=[]
    for day in days:
        try:
            origin=brussels_cutoff_timestamp(pd.Timestamp(day).date());h=forecast_horizon_hours(day)
            future=pd.date_range(origin,periods=h,freq='h',name='timestamp')
            past=pd.date_range(origin-pd.Timedelta(hours=2208),periods=2208,freq='h',name='timestamp')
            context=pd.DataFrame({k:s.reindex(past) for k,s in series.items()}).reset_index()
            context,quality=repair_context(context,expected_end=past[-1],context_hours=2208)
            weather_past=w.reindex(past);weather_future=w.reindex(future)
            if weather_past.isna().any().any() or weather_future.isna().any().any():
                raise ValueError('Missing archived weather covariates')
            tso=series['tso_forecast_mw'].reindex(future)
            if not np.isfinite(tso).all():raise ValueError('Incomplete TSO delivery grid')
            features=build_feature_frame(tso.rename('tso_forecast_mw').reset_index(),weather_future.reset_index())
            context=build_feature_frame(context,weather_past.reset_index())
            prepared[day]=(context,features,quality,series['actual_mw'].reindex(future).to_numpy())
        except Exception as exc:
            issues.append(dict(zone=zone,target=target,run_date=day,stage='prepare',error=str(exc)))
    return prepared,issues


def prepare_scope(inputs,zone,target,days):
    days=list(days)
    try:
        return _prepare_scope(inputs,zone,target,days)
    except Exception as exc:
        return {},[dict(zone=zone,target=target,run_date=day,stage='prepare',error=str(exc)) for day in days]


def predict_batch(adapter,key,target,covs,chunk):
    h=len(chunk[0][1]);contexts=[p.actual_mw.to_numpy(np.float32) for p,f in chunk]
    if key in ('t0_gamma_online','timestral_online'):
        arrays=[np.concatenate([p[covs].to_numpy(np.float32),f[covs].to_numpy(np.float32)]).T for p,f in chunk]
        return adapter.predict_batch(contexts,arrays,horizon=h)
    if key=='timesfm3_online':
        cleaned=[adapter._clean_covariates(p,f,covs) for p,f in chunk]
        arrays=[adapter._covariate_array(p,f,c) for p,f,c in cleaned]
        results=list(adapter._pipeline().predict_batch(contexts=contexts,horizon=h,past_future_covariates=arrays,
                        ts_ids=[f'{target}:{i}' for i in range(len(chunk))],make_positive=True,use_znorm=True))
        return np.stack([adapter._extract_forecast(r,h) for r in results])
    if key=='chronos2_online':
        pasts=[];futures=[]
        for i,(p,f) in enumerate(chunk):
            p=p[['timestamp','actual_mw',*covs]].copy().rename(columns={'actual_mw':'target'})
            f=f[['timestamp',*covs]].copy()
            for col in covs:
                p[col]=p[col].interpolate(limit_direction='both').ffill().bfill()
                f[col]=f[col].interpolate(limit_direction='both').ffill().bfill()
            pasts.append(p.assign(id=str(i)));futures.append(f.assign(id=str(i)))
        result=adapter._pipeline().predict_df(pd.concat(pasts,ignore_index=True),future_df=pd.concat(futures,ignore_index=True),
                    prediction_length=h,id_column='id',timestamp_column='timestamp',target='target',
                    quantile_levels=[.1,.5,.9],freq='h',validate_inputs=False)
        col='0.5' if '0.5' in result else 'median'
        return np.stack([result[result.id.eq(str(i))].sort_values('timestamp')[col].to_numpy(float) for i in range(len(chunk))])
    return np.stack([adapter.predict(f,target,context=p,covariates=covs) for p,f in chunk])


def replay(source,inputs,output,start,end,batch_size=8,selected_scopes=None,models=None):
    source,output=Path(source).resolve(),staging_path(output)
    if output==source or output.is_relative_to(source):raise ValueError('Use isolated staging')
    original=ForecastStore(source).read_forecasts();compact=source/'dashboard/forecasts_recent.parquet'
    if compact.exists():original=pd.concat([original,pd.read_parquet(compact)],ignore_index=True)
    original.run_date=original.run_date.astype(str)
    plan=original[original.run_date.between(start,end)][['run_date','zone','target','model']].drop_duplicates()
    registry=model_registry();enabled=zones();store=ForecastStore(output);audit=[];issues=[]
    adapters={};started=time.monotonic()
    for (zone,target),scope in plan.groupby(['zone','target'],sort=True):
        if zone not in enabled or target not in enabled[zone]['targets'] or (selected_scopes and (zone,target) not in selected_scopes):continue
        completed={day:complete_forecast_groups(store.forecast_path(day)) for day in scope.run_date.unique()}
        pending_days=sorted({r.run_date for r in scope.itertuples() if r.model in registry
            and registry[r.model].get('enabled') and not registry[r.model].get('pipeline_generated')
            and r.model!='tabpfn_online' and (not models or r.model in models)
            and (zone,target,r.model) not in completed[r.run_date]})
        if not pending_days:
            continue
        prepared,errors=prepare_scope(inputs,zone,target,pending_days);issues.extend(errors)
        for key,model_plan in scope.groupby('model',sort=True):
            cfg=registry.get(key,{})
            if key=='tabpfn_online' or not cfg.get('enabled') or cfg.get('pipeline_generated') or not model_supports_scope(cfg,zone,target) or (models and key not in models):continue
            if key not in adapters:adapters[key]=make_adapter(cfg)
            adapter=adapters[key];case,covs=covariates_for(target,zone,key)
            pending=[day for day in sorted(model_plan.run_date) if day in prepared and (zone,target,key) not in completed[day]]
            for h in sorted({len(prepared[d][1]) for d in pending}):
                days=[d for d in pending if len(prepared[d][1])==h]
                for offset in range(0,len(days),batch_size):
                    dates=days[offset:offset+batch_size];chunk=[prepared[d][:2] for d in dates];before=time.monotonic()
                    try:
                        predictions=np.asarray(predict_batch(adapter,key,target,covs,chunk),dtype=float)
                        if predictions.shape!=(len(dates),h) or not np.isfinite(predictions).all():raise ValueError('Invalid native forecast shape/values')
                        for day,pred in zip(dates,predictions,strict=True):
                            context,future,quality,truth=prepared[day]
                            frame=pd.DataFrame(dict(run_date=day,run_at=datetime.now(UTC).isoformat(),source='historical_mock_replay',
                                forecast_window='next_day_midnight_v1',input_vintage='retrospective_archived_revisions',
                                inference_device=getattr(adapter,'device',getattr(adapter,'device_map','cpu')),zone=zone,target=target,
                                model=key,model_label=cfg.get('label',key),covariate_case=case,context_hours=2208,
                                timestamp=future.timestamp,forecast_mw=pred,tso_forecast_mw=future.tso_forecast_mw,
                                actual_mw=truth,context_start=context.timestamp.min(),context_end=context.timestamp.max(),
                                context_missing_hours=quality.missing_hours,context_imputed_hours=quality.imputed_hours,
                                context_imputation_method=quality.methods,data_quality=quality.quality))
                            store.append_forecasts(frame,day);completed[day].add((zone,target,key))
                        audit.append(dict(zone=zone,target=target,model=key,hours=h,windows=len(dates),seconds=time.monotonic()-before))
                    except Exception as exc:
                        issues.extend(dict(zone=zone,target=target,model=key,run_date=d,stage='predict',error=str(exc)) for d in dates)
                    pd.DataFrame(audit).to_csv(output/'batch_runtime.csv',index=False)
                    pd.DataFrame(issues,columns=['zone','target','model','run_date','stage','error']).to_csv(output/'batch_issues.csv',index=False)
                    print(zone,target,key,h,min(offset+len(dates),len(days)),'/',len(days),'total_seconds',round(time.monotonic()-started,1),flush=True)
    close_workers()
    return issues


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','inputs','output','start','end']:p.add_argument('--'+name,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--scopes');p.add_argument('--models')
    a=p.parse_args();kwargs=vars(a).copy();scopes=kwargs.pop('scopes');models=kwargs.pop('models')
    kwargs['selected_scopes']={tuple(s.split(':')) for s in scopes.split(',')} if scopes else None
    kwargs['models']=set(models.split(',')) if models else None
    issues=replay(**kwargs)
    raise SystemExit(1 if issues else 0)
