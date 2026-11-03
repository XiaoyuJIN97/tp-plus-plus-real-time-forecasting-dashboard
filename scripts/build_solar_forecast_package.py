"""Rebuild shareable DE weather-point and saved-dashboard solar figures.

PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-solar python scripts/build_solar_forecast_package.py
Figures use a frozen saved-forecast snapshot, not the offline weather-point study.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from rt_forecast_dashboard.time_utils import is_complete_forecast_window, forecast_end_timestamp

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'docs/solar_forecast_package_2026-11-03'
MODELS = {'tso_reference':'TSO forecast','ensemble_selected':'Ensemble','timestral_online':'Timestral',
          'timesfm3_online':'TimesFM3','t0_gamma_online':'T0 Gamma','xgboost_online':'XGBoost',
          'chronos2_online':'Chronos2','persistence':'Persistence'}
ZONES = ['BE','DE','FR','ES','PT','DK1','DK2']
BLUE='#2563eb'; ORANGE='#ec8b23'; INK='#172b43'; GREY='#64748b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,
                     'axes.titleweight':'bold','savefig.facecolor':'white'})


def save(fig,path):
    for suffix in ('png','pdf','svg'):
        target=path.with_suffix('.'+suffix)
        fig.savefig(target,dpi=220,facecolor='white')
        if suffix=='svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def complete_panel(frame, models):
    """Exact full windows, shared forecasts and one frozen truth per timestamp."""
    pivot=frame[frame.model.isin(models)].pivot(index=['zone','run_date','timestamp'],columns='model',values='forecast_mw')
    for model in models:
        if model not in pivot: pivot[model]=np.nan
    truth=frame[frame.model.eq('tso_reference')].set_index(['zone','run_date','timestamp']).actual_mw
    panel=pivot[list(models)].join(truth.rename('truth')).reset_index()
    return panel.groupby(['zone','run_date'],group_keys=False).filter(
        lambda g:is_complete_forecast_window(g.timestamp,g.run_date.iloc[0])
        and np.isfinite(g[[*models,'truth']].to_numpy(float)).all())


def metrics(part,model):
    errors=part[model]-part.truth
    absolute=errors.abs().sum(); base=(part.persistence-part.truth).abs().sum()
    energy=part.truth.sum()
    return dict(runs=part.run_date.nunique(),hours=len(part),first_origin=part.run_date.min(),last_origin=part.run_date.max(),
                first_delivery_utc=part.timestamp.min().isoformat(),last_delivery_utc=part.timestamp.max().isoformat(),
                mae_mw=errors.abs().mean(),rmse_mw=np.sqrt(np.mean(errors**2)),wape_pct=100*absolute/energy,
                bias_pct=100*errors.sum()/energy,skill_vs_persistence_pct=100*(1-absolute/base))


def draw_map(out):
    points=json.loads((out/'de_solar_weather_points.json').read_text())['points']
    fig,(ax,table)=plt.subplots(1,2,figsize=(12,8),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.06,right=.97,top=.84,bottom=.12,wspace=.10)
    for f in json.loads((out/'map_boundaries.geojson').read_text())['features']:
        geometry=f['geometry']; polygons=geometry['coordinates'] if geometry['type']=='MultiPolygon' else [geometry['coordinates']]
        for polygon in polygons:
            ring=np.asarray(polygon[0]); german=f['properties']['ADMIN']=='Germany'
            ax.add_patch(Polygon(ring,facecolor='#e3effb' if german else '#f4f5f7',edgecolor='#718399' if german else '#cad2dc',lw=1.1 if german else .6,zorder=1))
    for p in points:
        ax.scatter(p['longitude'],p['latitude'],s=240,color=ORANGE,edgecolor='white',linewidth=1.7,zorder=3)
        ax.text(p['longitude'],p['latitude'],str(p['point']),ha='center',va='center',color='white',weight='bold',zorder=4)
    ax.set(xlim=(5.4,15.5),ylim=(47.0,55.5),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    ax.set_aspect(1/np.cos(np.deg2rad(51))); ax.grid(alpha=.17,zorder=0)
    ax.set_title('Currently deployed request locations',loc='left',fontsize=12,pad=15)
    table.axis('off')
    table.text(0, .91, 'Point    Latitude °N    Longitude °E',weight='bold',fontsize=13)
    for i,p in enumerate(points):
        table.text(0,.81-i*.105,f"  {p['point']}          {p['latitude']:.2f}              {p['longitude']:.2f}",fontsize=14)
    table.text(0,.31,'Selection lineage',weight='bold',fontsize=13)
    table.text(0,.255,'Capacity-weighted clustering (k = 4)\nOriginal sample: 16 solar assets, 1,619.711 MW\nCentroids → ERA5 → nominal GFS grid',fontsize=11,linespacing=1.8,va='top')
    table.text(0,.075,'The collector uses the coordinates above.\nThey differ slightly from the original planned grids.\nThe newer MaStR-based candidates are not deployed.',fontsize=10,color=GREY,linespacing=1.7,va='top')
    fig.suptitle('Germany solar forecast | selected weather points',x=.06,ha='left',fontsize=20,weight='bold',y=.96)
    fig.text(.06,.90,'Four input locations retained in the online configuration · snapshot 3 November 2026',fontsize=11,color=GREY)
    fig.text(.06,.035,'Coordinates: deployed dashboard + collector configuration. Boundaries: Natural Earth (public domain).',fontsize=9,color=GREY)
    save(fig,out/'01_de_solar_weather_points_map')


def build(out):
    source=out/'solar_forecasts_snapshot.parquet'
    frame=pd.read_parquet(source)
    frame.timestamp=pd.to_datetime(frame.timestamp,utc=True);frame.run_date=frame.run_date.astype(str)
    meta=json.loads((out/'source_manifest.json').read_text())
    cutoff=pd.Timestamp(meta['snapshot_generated_at_utc'])
    frame=frame[frame.run_date.map(lambda d:forecast_end_timestamp(d)<=cutoff)].copy()
    assert not frame.duplicated(['zone','run_date','timestamp','model']).any()
    panel=complete_panel(frame,MODELS)
    rows=[]
    for zone,g in panel.groupby('zone'):
        tso_mae=(g.tso_reference-g.truth).abs().mean()
        for model,label in MODELS.items():
            rows.append(dict(zone=zone,model=model,label=label,**metrics(g,model),rmae=metrics(g,model)['mae_mw']/tso_mae))
    scores=pd.DataFrame(rows);scores.to_csv(out/'dashboard_solar_metrics.csv',index=False)
    panel[['zone','run_date','timestamp','truth']].to_parquet(out/'dashboard_scoring_truth.parquet',index=False)
    panel[['zone','run_date']].drop_duplicates().to_csv(out/'dashboard_shared_origins.csv',index=False)
    de=scores[scores.zone.eq('DE')].sort_values('mae_mw')
    order=['TSO forecast','Ensemble','Timestral','TimesFM3','T0 Gamma','XGBoost','Chronos2','Persistence']
    heat=scores.pivot(index='label',columns='zone',values='rmae').reindex(index=order,columns=ZONES)
    fig,(ax,hm)=plt.subplots(1,2,figsize=(17,9.3),gridspec_kw={'width_ratios':[1,1.38]})
    fig.subplots_adjust(left=.10,right=.94,top=.79,bottom=.26,wspace=.52)
    ax.barh(de.label,de.mae_mw,color=[ORANGE if m=='ensemble_selected' else BLUE if m=='tso_reference' else '#a9bdd7' for m in de.model],height=.65)
    ax.invert_yaxis();ax.set_xlim(0,de.mae_mw.max()*1.24);ax.set_xlabel('Mean absolute error (MW) · lower is better');ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    for i,r in enumerate(de.itertuples()): ax.text(r.mae_mw+20,i,f'{r.mae_mw:,.0f}',va='center',fontsize=11)
    ax.set_title('Germany / Luxembourg aggregate',loc='left',pad=35,fontsize=14)
    ax.text(0,1.035,f"{de.runs.iloc[0]} shared origins · {de.hours.iloc[0]:,} delivery hours per model",transform=ax.transAxes,fontsize=10,color=GREY)
    image=hm.imshow(heat.to_numpy(),aspect='auto',cmap='RdYlGn_r',norm=TwoSlopeNorm(vmin=0.,vcenter=1.,vmax=4.5))
    hm.set_xticks(range(len(ZONES)),ZONES);hm.set_yticks(range(len(order)),order)
    hm.set_title('All dashboard solar series | error relative to TSO',loc='left',pad=35,fontsize=14)
    counts=scores.groupby('zone').runs.first().reindex(ZONES)
    hm.text(0,1.035,'Shared origins within each zone: '+', '.join(f'{z} {int(counts[z])}' for z in ZONES),transform=hm.transAxes,fontsize=8.7,color=GREY)
    for i in range(len(order)):
        for j in range(len(ZONES)):
            val=heat.iloc[i,j]
            hm.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=10,color='white' if val>3.8 else INK)
    cb=fig.colorbar(image,ax=hm,fraction=.045,pad=.035,extend='max');cb.set_label('rMAE = model MAE / TSO MAE')
    fig.suptitle('Solar forecasting performance | saved online dashboard results',x=.055,ha='left',fontsize=21,weight='bold',y=.96)
    fig.text(.055,.905,'Full midnight-ending windows · exact shared origins within each zone · one common observed truth per delivery interval',fontsize=11,color=GREY)
    fig.text(.055,.055,'rMAE < 1 beats TSO; 1 matches TSO. Colors saturate above 4.5; cell values are exact to two decimals.\nPeriods differ by zone: see dashboard_solar_metrics.csv. Includes archived-input mock replays; this is not the weather-point ablation study.\nContext: 2,208 hours. Forecast window: 18:00 Brussels → midnight after the following day (29/30/31 hours). Snapshot: 3 Nov 2026.',fontsize=10,color=GREY,linespacing=1.65)
    save(fig,out/'02_dashboard_solar_performance')
    # Cross-TSO comparisons require identical dates across areas, independent of ML model availability.
    tso=complete_panel(frame,['tso_reference','persistence'])
    rows=[]; origins=[]
    comparisons=[('national',['BE','DE','FR','ES','PT']),('all_areas',ZONES)]
    for label,zones in comparisons:
        days=sorted(set.intersection(*(set(tso[tso.zone.eq(z)].run_date) for z in zones)))
        for z in zones:
            part=tso[tso.zone.eq(z)&tso.run_date.isin(days)]
            rows.append(dict(comparison=label,zone=z,**metrics(part,'tso_reference')))
        origins.extend(dict(comparison=label,run_date=d) for d in days)
    comparison=pd.DataFrame(rows);comparison.to_csv(out/'tso_solar_metrics.csv',index=False)
    pd.DataFrame(origins).to_csv(out/'tso_shared_origins.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,8.6))
    fig.subplots_adjust(left=.12,right=.95,top=.77,bottom=.30,wspace=.42)
    names={'BE':'Belgium','DE':'Germany + LU','FR':'France','ES':'Spain','PT':'Portugal','DK1':'Denmark DK1','DK2':'Denmark DK2'}
    for ax,(key,zones),title in zip(axes,comparisons,['Five national / market aggregates','All seven dashboard areas']):
        part=comparison[comparison.comparison.eq(key)].sort_values('wape_pct')
        ax.barh([names[z] for z in part.zone],part.wape_pct,color=[BLUE if z=='DE' else '#a9bdd7' for z in part.zone],height=.6)
        ax.invert_yaxis();ax.set_xlim(0,part.wape_pct.max()*1.22);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
        ax.set_xlabel('WAPE (%) · lower is better');ax.set_title(title,loc='left',pad=48,fontsize=14)
        head=part.iloc[0]
        ax.text(0,1.045,f"{head.first_origin} → {head.last_origin}\n{head.runs} shared origins · {head.hours:,} delivery hours per area",transform=ax.transAxes,fontsize=10,color=GREY,linespacing=1.5)
        for i,r in enumerate(part.itertuples()):ax.text(r.wape_pct+.25,i,f'{r.wape_pct:.1f}%',va='center',fontsize=12)
    fig.suptitle('Solar TSO forecast performance | matched delivery windows',x=.055,ha='left',fontsize=21,weight='bold',y=.96)
    fig.text(.055,.90,'Size-normalized comparison of the TSO benchmark curves stored by our dashboard',fontsize=11,color=GREY)
    fig.text(.055,.055,'WAPE = 100 × sum(|forecast − actual|) / sum(actual). Includes all delivery hours, including night; no division by individual zero-output hours.\nEach panel uses identical complete origin dates across its areas; the two panels use different intersections.\nDE follows the collector’s DE/LU label; historical fallback domain provenance is incomplete. DK1/DK2 are separate areas.\nIncludes archived-input replays and current stored revisions; these figures do not establish an intrinsic ranking of operator forecasting skill.',fontsize=10,color=GREY,linespacing=1.65)
    save(fig,out/'03_tso_solar_performance')
    draw_map(out)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(files,indent=2)+'\n')
    print('DE scores:',de[['label','mae_mw','rmae','runs','first_origin','last_origin']].to_string(index=False))
    print('TSO scores:',comparison[['comparison','zone','wape_pct','runs','first_origin','last_origin']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=PACKAGE)
    build(parser.parse_args().package)
