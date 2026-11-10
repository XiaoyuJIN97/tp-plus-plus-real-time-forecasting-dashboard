"""Stage audited weather replacements; preserve genuine live predictions exactly."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import is_complete_forecast_window,brussels_cutoff_timestamp

KEYS=['run_date','zone','target','model']
HOUR_KEYS=KEYS+['timestamp']


def stage(source,replay,audit,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists() or output==source or output.is_relative_to(source):
        raise ValueError('Use a new isolated staging directory')
    before=ForecastStore(source).read_forecasts();before.run_date=before.run_date.astype(str)
    original=before[before.source.str.contains('mock|replay',case=False,regex=True)]
    incoming=ForecastStore(Path(replay)).read_forecasts();incoming.run_date=incoming.run_date.astype(str)
    allowed=original[KEYS].drop_duplicates()
    if len(incoming.merge(allowed,on=KEYS,how='inner'))!=len(incoming):
        raise ValueError('Replacement targets a non-mock group')
    evidence={};comparison=[];validated=[]
    original_indices = before.groupby(KEYS).indices
    for key,part in incoming.groupby(KEYS):
        if not is_complete_forecast_window(part.timestamp,key[0]) or not np.isfinite(part.forecast_mw).all():
            raise ValueError('Incomplete/nonfinite replacement')
        if not part.weather_vintage_policy.eq('weather_collection_asof_v1').all():
            raise ValueError('Replacement lacks weather availability proof')
        proof=json.loads(part.weather_vintage_provenance.iloc[0]);cutoff=brussels_cutoff_timestamp(pd.Timestamp(key[0]).date())
        if pd.Timestamp(proof['cutoff_utc'])!=cutoff or not proof['snapshots']:
            raise ValueError('Invalid cutoff/provenance')
        if any(pd.Timestamp(p['available_at'])>cutoff for p in proof['snapshots']):
            raise ValueError('Replacement contains later-collected weather')
        old=before.iloc[original_indices[key]]
        joined=part.merge(old[['timestamp','forecast_mw','actual_mw']],on='timestamp',suffixes=('','_old'),validate='one_to_one')
        if len(joined)!=len(part):raise ValueError('Replacement delivery grid differs')
        delta=joined.forecast_mw.to_numpy()-joined.forecast_mw_old.to_numpy()
        changed=not np.allclose(delta,0,atol=1e-8,rtol=0)
        comparison.append(dict(zip(KEYS,key),changed=changed,max_absolute_change_mw=float(np.max(np.abs(delta))),rows=len(part)))
        # Preserve the established common scoring truth; weather repair does not revise it.
        joined['actual_mw']=joined.pop('actual_mw_old');joined=joined.drop(columns='forecast_mw_old')
        serialized=json.dumps(proof,sort_keys=True)
        proof_id=hashlib.sha256(serialized.encode()).hexdigest();evidence[proof_id]=proof
        joined=joined.drop(columns='weather_vintage_provenance')
        joined['weather_provenance_id']=proof_id
        joined['weather_vintage_status']='corrected_weather_asof'
        validated.append(joined)
    plan=pd.read_csv(audit,dtype={'run_date':str})
    eligible=plan[plan.status.eq('weather_available')][KEYS].drop_duplicates()
    if len(incoming.merge(eligible,on=KEYS,how='inner'))!=len(incoming):
        raise ValueError('Replacement is not eligible in the frozen audit')
    output.mkdir(parents=True)
    for folder in ['forecasts','dashboard']:
        if (source/folder).exists():shutil.copytree(source/folder,output/folder)
    store=ForecastStore(output)
    # Daily score caching is intentional; a historical prediction replacement
    # must explicitly invalidate it even when realized coverage is unchanged.
    (store.dashboard_dir()/'leaderboard_daily_status.json').unlink(missing_ok=True)
    corrected=pd.concat(validated,ignore_index=True) if validated else incoming.iloc[:0]
    for day,part in corrected.groupby('run_date'):store.append_forecasts(part,day)
    after=store.read_forecasts();after.run_date=after.run_date.astype(str)
    live=before[before.source.eq('online')][HOUR_KEYS+['forecast_mw']]
    result=after[after.source.eq('online')][HOUR_KEYS+['forecast_mw']]
    check=live.merge(result,on=HOUR_KEYS,how='outer',suffixes=('_before','_after'),indicator=True,validate='one_to_one')
    if not check._merge.eq('both').all() or not np.allclose(check.forecast_mw_before,check.forecast_mw_after,atol=1e-8,rtol=1e-12):
        raise ValueError('Genuine online predictions changed')
    pd.DataFrame(comparison).to_csv(output/'weather_prediction_comparison.csv',index=False)
    (output/'weather_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    report=dict(replacement_groups=len(comparison),changed_groups=sum(r['changed'] for r in comparison),
                genuine_online_prediction_rows_preserved=len(live),prediction_absolute_tolerance_mw=1e-8,full_input_certification=False)
    (output/'weather_repair_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','replay','audit','output']:p.add_argument('--'+name,required=True)
    stage(**vars(p.parse_args()))
