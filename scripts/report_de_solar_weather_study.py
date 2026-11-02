"""Score paired weather-point replays on shared, fully observed online windows."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import forecast_horizon_hours
from rt_forecast_dashboard.conformal import conformal_intervals,interval_leaderboard
from scripts.de_solar_weather_study import MODELS,expected_grid

LABELS={'tso_reference':'TSO','persistence':'Persistence','chronos2_online':'Chronos2','timesfm3_online':'TimesFM3',
        'xgboost_online':'XGBoost','t0_gamma_online':'T0 Gamma','timestral_online':'Timestral','ensemble_selected':'Ensemble'}


def shared_windows(frames):
    complete={};expected=set(LABELS)
    for arm,d in frames.items():
        d['timestamp']=pd.to_datetime(d.timestamp,utc=True);valid=[]
        for day,g in d.groupby('run_date'):
            good=True
            for model in expected:
                x=g[g.model.eq(model)].sort_values('timestamp')
                if (len(x)!=forecast_horizon_hours(day) or not pd.DatetimeIndex(x.timestamp).equals(expected_grid(day))
                    or not np.isfinite(x[['forecast_mw','actual_mw','tso_forecast_mw']].to_numpy(float)).all()):good=False;break
            if good:valid.append(str(day))
        complete[arm]=set(valid)
    return sorted(set.intersection(*complete.values()))


def bootstrap_delta(daily,draws=2000):
    """Paired moving-block bootstrap: seven adjacent eligible forecast origins."""
    a=daily.sort_values('run_date');n=len(a)
    if n==0:return np.nan,np.nan
    b=a.baseline_abs_sum.to_numpy();c=a.candidate_abs_sum.to_numpy();h=a.hours.to_numpy()
    rng=np.random.default_rng(42);block=min(7,n);starts=rng.integers(0,n,size=(draws,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
    values=(c[idx].sum(axis=1)-b[idx].sum(axis=1))/h[idx].sum(axis=1)
    return np.quantile(values,[.025,.975])


def metrics(d):
    rows=[]
    d=d.copy();d['abs_error']=(d.forecast_mw-d.actual_mw).abs();d['sq_error']=(d.forecast_mw-d.actual_mw)**2
    for (arm,model),g in d.groupby(['arm','model']):
        rows.append(dict(arm=arm,model=model,model_label=LABELS[model],mae_mw=g.abs_error.mean(),rmse_mw=np.sqrt(g.sq_error.mean()),
                         generating_hours_mae_mw=g.loc[g.actual_mw>100,'abs_error'].mean(),observations=len(g),runs=g.run_date.nunique()))
    out=pd.DataFrame(rows);ref=out[out.model.eq('tso_reference')].set_index('arm').mae_mw
    out['rmae_vs_tso']=[r.mae_mw/ref[r.arm] for r in out.itertuples()];return out


def report(root,output):
    output.mkdir(parents=True,exist_ok=True)
    frames={a:ForecastStore(root/'predictions'/a).read_forecasts() for a in ['baseline','candidate']}
    for d in frames.values():d['run_date']=d.run_date.astype(str)
    days=shared_windows(frames)
    if not days:raise ValueError('No shared complete model windows')
    parts=[d[d.run_date.isin(days)&d.model.isin(LABELS)].assign(arm=a) for a,d in frames.items()]
    d=pd.concat(parts,ignore_index=True);d.timestamp=pd.to_datetime(d.timestamp,utc=True)
    keys=['run_date','timestamp','model'];paired=parts[0].merge(parts[1],on=keys,suffixes=('_baseline','_candidate'),validate='one_to_one')
    assert np.allclose(paired.actual_mw_baseline,paired.actual_mw_candidate,rtol=0,atol=1e-8)
    assert np.allclose(paired.tso_forecast_mw_baseline,paired.tso_forecast_mw_candidate,rtol=0,atol=1e-8)
    control=paired.model.isin(['tso_reference','persistence'])
    assert np.allclose(paired.loc[control,'forecast_mw_baseline'],paired.loc[control,'forecast_mw_candidate'],rtol=0,atol=1e-8),'Weather-independent controls changed'
    m=metrics(d);m.to_csv(output/'metrics.csv',index=False)
    comparison=m[m.arm.eq('baseline')].drop(columns='arm').merge(m[m.arm.eq('candidate')].drop(columns='arm'),on=['model','model_label'],suffixes=('_baseline','_candidate'))
    comparison['mae_change_pct']=100*(comparison.mae_mw_candidate/comparison.mae_mw_baseline-1)
    paired['baseline_abs_sum']=(paired.forecast_mw_baseline-paired.actual_mw_baseline).abs()
    paired['candidate_abs_sum']=(paired.forecast_mw_candidate-paired.actual_mw_candidate).abs()
    daily=paired.groupby(['model','run_date']).agg(baseline_abs_sum=('baseline_abs_sum','sum'),candidate_abs_sum=('candidate_abs_sum','sum'),hours=('timestamp','size')).reset_index()
    daily.to_csv(output/'paired_daily_errors.csv',index=False)
    for index,row in comparison.iterrows():
        g=daily[daily.model.eq(row.model)];lo,hi=bootstrap_delta(g)
        comparison.loc[index,'delta_mae_ci_low_mw']=lo;comparison.loc[index,'delta_mae_ci_high_mw']=hi
        diff=g.candidate_abs_sum-g.baseline_abs_sum
        comparison.loc[index,'candidate_win_pct']=100*((diff<0).mean()+.5*(diff==0).mean())
    comparison.to_csv(output/'comparison.csv',index=False)
    d['origin_month']=d.run_date.str[:7]
    monthly=pd.concat([metrics(g).assign(origin_month=month) for month,g in d.groupby('origin_month')],ignore_index=True)
    monthly.to_csv(output/'monthly_metrics.csv',index=False)
    for arm in ['baseline','candidate']:
        bands=conformal_intervals(d[d.arm.eq(arm)].copy())
        interval_leaderboard(bands,.8).assign(arm=arm).to_csv(output/f'interval_metrics_{arm}.csv',index=False)
    quality=pd.read_csv(root/'context_quality.csv');quality.to_csv(output/'context_quality.csv',index=False)
    missing=pd.read_csv(root/'missing_online_tso.csv');missing.to_csv(output/'excluded_origins.csv',index=False)
    for p in ['preperiod_points.csv','current_points.csv','selection_audit.json','configuration_hashes.json','inference_timing.csv']:
        (output/p).write_bytes((root/p).read_bytes())
    manifest={str(p.relative_to(root)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (root/'inputs').rglob('*') if p.is_file()}
    (output/'input_hashes.json').write_text(json.dumps(manifest,indent=2))
    summary=dict(first_origin=min(days),last_origin=max(days),origins=len(days),observations_per_model=int(m.observations.iloc[0]),
                 first_delivery=str(d.timestamp.min()),last_delivery=str(d.timestamp.max()),
                 online_configuration_unchanged=True,weather_points_per_arm=4,controls_identical=True,
                 note='Paired retrospective weather ablation. Stitched historical forecasts do not reproduce issue-time 18:00 forecast vintages; absolute scores are not production guarantees.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    plots(root,output,comparison)
    records=comparison.set_index('model')
    lines=['# German solar weather-point comparison — July–November 2026','',
      f"Paired evaluation: **{len(days)} forecast origins**, {min(days)} through {max(days)}; **{int(m.observations.iloc[0]):,} hourly predictions per model**. Last delivery: {summary['last_delivery']} UTC.",'',
      'Only weather-point locations changed. Four points, the 2,208-hour context, 18:00 Brussels cutoff, next-day-midnight horizon, CPU model adapters/checkpoints, nine solar covariates including TSO, and the configured median ensemble were preserved. Original stored online TSO delivery curves and a common frozen actual history were used for both arms.','',
      '| Model | Original points MAE (MW) | New points MAE (MW) | Change | New/original win rate |',
      '|---|---:|---:|---:|---:|']
    for key in LABELS:
        r=records.loc[key];lines.append(f"| {LABELS[key]} | {r.mae_mw_baseline:.1f} | {r.mae_mw_candidate:.1f} | {r.mae_change_pct:+.1f}% | {r.candidate_win_pct:.1f}% |")
    lines+=['','Negative change means lower error. Win rate counts ties as half wins. The primary metric includes nighttime hours, matching the online leaderboard. Generating-hours MAE (actual >100 MW), RMSE, monthly results, and paired seven-origin block-bootstrap intervals are supplied as CSVs.','',
      '![Paired accuracy](accuracy.png)','','## Proposed four points','',
      pd.read_csv(root/'preperiod_points.csv')[['point','gfs_lat','gfs_lon','cluster_share']].to_markdown(index=False),'',
      'These points come from capacity-weighted K-means (k=4) on the full public MaStR solar inventory, including rooftop installations. Exact private coordinates were replaced by public postcode centroids where needed. Capacity shares describe selection clusters; they are not new weights applied to the forecast models.','',
      'The current-fleet and pre-period-fleet selections snapped to identical weather grids. The primary selection filters commission and registration dates to 23 July 2026. The underlying registry export is from 2 November and may contain later metadata corrections, so it is not a certified as-issued July inventory.','',
      '![Weather points](weather_points.png)','','## Interpretation and limits','',
      'Historical Open-Meteo forecasts are stitched from short-lead model runs. Both arms use the same endpoint, automatic best-match policy, variables and retrieval method, but these are not the original day-ahead weather vintages. The paired differences isolate point selection within this reconstruction; absolute errors should not be presented as live operational performance.','',
      'Two origins without complete stored online TSO delivery curves are excluded from both arms. Actual observations are never imputed for scoring. Context repair retains the production policy and is audited in context_quality.csv. Forecast outcomes were not used to choose points.','',
      'All outputs are offline research artifacts; the online collector and dashboard point configuration were not replaced.','',
      'Sources: [MaStR / Bundesnetzagentur](https://www.marktstammdatenregister.de/MaStR/Datendownload) (Datenlizenz Deutschland – Namensnennung 2.0); [GeoNames postcode centroids](https://download.geonames.org/export/zip/) (CC BY 4.0); [Open-Meteo historical forecasts](https://open-meteo.com/en/docs/historical-forecast-api) (weather data attribution).','']
    (output/'README.md').write_text('\n'.join(lines))
    print(comparison[['model_label','mae_mw_baseline','mae_mw_candidate','mae_change_pct']].to_string(index=False),flush=True)


def plots(root,out,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison=comparison[comparison.model.ne('persistence')].copy()
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(comparison));ax.bar(x-.2,comparison.mae_mw_baseline,.4,label='Original points');ax.bar(x+.2,comparison.mae_mw_candidate,.4,label='Capacity-based points')
    
    for i,r in enumerate(comparison.itertuples()):
        ax.text(i,max(r.mae_mw_baseline,r.mae_mw_candidate)+12,f'{r.mae_change_pct:+.1f}%',ha='center',fontsize=9,color='#9c2534' if r.mae_change_pct>0 else '#247450')
    ax.set_ylim(0,max(comparison.mae_mw_baseline.max(),comparison.mae_mw_candidate.max())*1.16)
    ax.set_xticks(x,comparison.model_label,rotation=25,ha='right');ax.set_ylabel('MAE (MW)');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=180);plt.close(fig)
    grid=pd.read_parquet(root/'current_capacity_grid.parquet');old=pd.read_csv(root/'baseline_points.csv') if (root/'baseline_points.csv').exists() else pd.read_csv('config/selected_weather_points_for_tp_v01_4p.csv');old=old[(old.country=='DE')&(old.type=='solar')];new=pd.read_csv(root/'preperiod_points.csv')
    fig,ax=plt.subplots(figsize=(8,9));im=ax.scatter(grid.longitude,grid.latitude,c=np.log10(grid.net_mw+.01),s=4,cmap='YlOrBr',alpha=.65,rasterized=True)
    ax.scatter(old.gfs_lon,old.gfs_lat,c='#c82534',s=100,marker='x',linewidths=2,label='Original points');ax.scatter(new.gfs_lon,new.gfs_lat,c='#1368b5',s=110,marker='*',edgecolors='white',linewidths=.7,label='Capacity-based points')
    for r in new.itertuples():ax.annotate(str(r.point),(r.gfs_lon,r.gfs_lat),xytext=(6,6),textcoords='offset points',color='#1368b5',weight='bold')
    ax.set_aspect(1/np.cos(np.radians(51)));ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)');ax.set_title('German PV capacity and four-point weather selection');ax.legend(loc='lower left');fig.colorbar(im,ax=ax,label='log10 net capacity per coarse cell (MW + 0.01)',shrink=.7);fig.tight_layout();fig.savefig(out/'weather_points.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.root.resolve(),a.output.resolve())
