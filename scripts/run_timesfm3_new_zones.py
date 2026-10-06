from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / 'notebooks' / 'timesfm3_offline_2021_2025_8bz.ipynb'

def load_notebook_runtime():
    nb=json.loads(NOTEBOOK.read_text())
    ns={'display': lambda *args, **kwargs: None}
    for idx in [2,3,5,7,8,10,12,14,16]:
        source=''.join(nb['cells'][idx].get('source',[]))
        exec(compile(source, f'{NOTEBOOK.name}:cell{idx}', 'exec'),ns)
    return ns

def run_zone_target_batched(ns, zone, target, adapter, force=False, batch_size=128):
    import numpy as np, time as pytime
    out_path=ns['FORECAST_DIR']/f'timesfm3_{zone}_{target}_2021_2025.csv'
    if out_path.exists() and not force:
        print(f'Skip existing {zone} {target}: {out_path}',flush=True)
        return pd.read_csv(out_path)
    data=ns['load_zone_target_data'](zone,target,ns['TEST_YEARS'])
    report=ns['coverage_report'](data)
    if (report['rows']==0).any():
        print(f'Missing source for {zone} {target}\n{report.to_string(index=False)}',flush=True); return pd.DataFrame()
    case,covs=ns['covariates_for_timesfm3'](target,zone)
    if target in {'load', 'solar'}:
        cutoffs=[]
        for year in ns['TEST_YEARS']:
            cutoffs.extend(pd.date_range(f'{year}-01-01 18:00:00+00:00', f'{year}-12-31 18:00:00+00:00', freq='D').tolist())
    else:
        cutoffs=ns['daily_cutoffs_for_years'](ns['TEST_YEARS'])
    if ns['MAX_CUTOFFS_PER_ZONE_TARGET'] is not None: cutoffs=cutoffs[:ns['MAX_CUTOFFS_PER_ZONE_TARGET']]
    prepared=[]; issues=[]
    for cutoff in cutoffs:
        try:
            context,features,actual=ns['prepare_cutoff_frames'](data,cutoff)
            missing=[c for c in covs if c not in context.columns or c not in features.columns]
            if missing or len(context)<ns['MIN_CONTEXT_HOURS'] or len(features)<ns['HORIZON_HOURS']:
                raise ValueError(f'missing={missing}, context={len(context)}, horizon={len(features)}')
            context,features,valid=adapter._clean_covariates(context,features,covs)
            prepared.append((cutoff,context,features,actual,valid))
        except Exception as exc: issues.append({'zone':zone,'target':target,'cutoff_utc':cutoff,'issue':str(exc)})
    print(f'{zone} {target}: prepared={len(prepared)} issues={len(issues)}',flush=True)
    model=adapter._load_model(); rows=[]; started=pytime.time()
    for start in range(0,len(prepared),batch_size):
        chunk=prepared[start:start+batch_size]
        contexts=[x[1]['actual_mw'].to_numpy(dtype=np.float32) for x in chunk]
        cov_arrays=[adapter._make_covariate_array(x[1],x[2],x[4]) for x in chunk]
        results=list(model.predict_batch(contexts=contexts,horizon=ns['HORIZON_HOURS'],past_future_covariates=cov_arrays,ts_ids=[f'{zone}:{target}:{x[0]}' for x in chunk],make_positive=True,use_znorm=True))
        for (cutoff,context,features,actual,valid),result in zip(chunk,results,strict=True):
            yhat=adapter._extract_forecast(result,len(features))
            base=pd.DataFrame({'zone':zone,'target':target,'model':'timesfm3_offline','model_label':'TimesFM3 with dashboard covariates','covariate_case':case,'cutoff_utc':cutoff,'delivery_time_brussels':pd.to_datetime(features['timestamp'],utc=True).dt.tz_convert(ns['BRUSSELS_TZ']),'timestamp':pd.to_datetime(features['timestamp'],utc=True),'horizon':np.arange(ns['HORIZON_HOURS']),'forecast_mw':yhat,'tso_forecast_mw':features['tso_forecast_mw'].to_numpy(float),'context_hours':len(context),'context_start':pd.to_datetime(context['timestamp'],utc=True).min(),'context_end':pd.to_datetime(context['timestamp'],utc=True).max()}).merge(actual,on='timestamp',how='left')
            rows.extend([base,base.assign(model='tso_reference',model_label='TSO forecast',covariate_case='tso_forecast',forecast_mw=base['tso_forecast_mw'].to_numpy(float)),base.assign(model='persistence',model_label='Persistence',covariate_case='weekly_persistence' if target=='load' else 'daily_persistence',forecast_mw=ns['persistence_forecast'](context,features,target))])
        print(f'{zone} {target}: inferred {min(start+batch_size,len(prepared))}/{len(prepared)} elapsed={(pytime.time()-started)/60:.1f} min',flush=True)
    forecast=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if not forecast.empty: forecast.to_csv(out_path,index=False); print('Wrote',out_path,len(forecast),'rows',flush=True)
    if issues: pd.DataFrame(issues).to_csv(ns['FORECAST_DIR']/f'issues_{zone}_{target}_2021_2025.csv',index=False)
    return forecast

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zones',default='NL,DK1,DK2,ES,PT')
    parser.add_argument('--targets',default='load,solar,wind_onshore,wind_offshore')
    parser.add_argument('--years',default='2023,2024,2025')
    parser.add_argument('--max-cutoffs',type=int,default=None)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    ns=load_notebook_runtime()
    zones=[x.strip() for x in args.zones.split(',') if x.strip()]
    targets=[x.strip() for x in args.targets.split(',') if x.strip()]
    years=[int(x) for x in args.years.split(',')]
    out=REPO_ROOT/'notebooks'/'outputs'/'timesfm3_offline_new_zones'
    forecast_dir=out/'forecasts_by_zone_target'; csv_dir=out/'csv'
    forecast_dir.mkdir(parents=True,exist_ok=True); csv_dir.mkdir(parents=True,exist_ok=True)
    ns.update({'ZONES':zones,'TARGETS':targets,'TEST_YEARS':years,'MAX_CUTOFFS_PER_ZONE_TARGET':args.max_cutoffs,'OUTPUT_DIR':out,'FORECAST_DIR':forecast_dir,'CSV_DIR':csv_dir, 'WEATHER_ROOTS':[Path('/Users/xiaoyujin/Desktop/TP++/Load_forecast_new/weather_forecasts_4p_NL_DK_ES_PT'), Path('/Users/xiaoyujin/Desktop/TP++/Solar_forecast_tabpfn_new/outputs/solar_4p_candidate_points/weather_forecasts_4p_NL_DK1_DK2_ES_PT'), Path('/Users/xiaoyujin/Desktop/TP++/Wind_forecast_new/data/weather_forecasts_4p_new_countries')]})
    # Use the exact dashboard load selections, including without-TSO and TSO-only cases.
    # DK1's archived load-weather file contains temperature/humidity but no wind;
    # use its available temperature covariates while retaining the selected TSO input.
    load_cfg={'NL':('Solar',False),'DK1':('Temp',True),'DK2':('deg_proxy',True),'ES':('deg_proxy',True),'PT':('Hum',True)}
    def configured_covariates(target,zone):
        if target=='load':
            feature,include_tso=load_cfg[zone]
            covs=list(ns['LOAD_GROUPS'].get(feature, []))
            if include_tso: covs.append('tso_forecast_mw')
            return '+'.join(([feature] if feature else [])+(['tso_forecast'] if include_tso else [])),covs
        if target=='solar': return 'weather_4p+tso_forecast', [*ns['SOLAR_COVARIATES'],'tso_forecast_mw']
        return 'weather_4p+tso_forecast', [*ns['WIND_COVARIATES'],'tso_forecast_mw']
    ns['covariates_for_timesfm3']=configured_covariates
    adapter=ns['TimesFM3Adapter'](checkpoint_path='google/timesfm-3.0-pytorch',device='cpu',standardize_covariates=False,local_files_only=True)
    outputs=[]
    for zone in zones:
        for target in targets:
            if zone=='ES' and target=='wind_offshore':
                print('Skip ES wind_offshore: unavailable series',flush=True); continue
            frame=run_zone_target_batched(ns,zone,target,adapter,force=args.force,batch_size=args.batch_size)
            if not frame.empty: outputs.append(frame)
    if outputs:
        combined=pd.concat(outputs,ignore_index=True)
        combined.to_csv(csv_dir/'timesfm3_new_zones_forecasts_long.csv',index=False)
        ns['metric_summary'] = None
        print(combined.groupby(['zone','target','model']).size().to_string(),flush=True)
        print('rows',len(combined),flush=True)
if __name__=='__main__': main()
