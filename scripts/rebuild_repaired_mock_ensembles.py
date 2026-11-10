"""Recompute only historical ensemble outputs, never replace a live ensemble."""
from pathlib import Path
import argparse,json,shutil
import numpy as np,pandas as pd
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.ui.analytics import valid_online_forecasts
from rt_forecast_dashboard.pipeline.build_selected_ensembles import build_selected_ensembles


def rebuild(staged,scratch,first):
    staged=Path(staged).resolve();scratch=Path(scratch).resolve()
    if scratch==staged or scratch.is_relative_to(staged):raise ValueError('Use a separate scratch directory')
    if scratch.exists():raise ValueError('Use a clean ensemble work directory')
    shutil.copytree(staged/'forecasts',scratch/'forecasts')
    store=ForecastStore(staged);before=store.read_forecasts();k=['run_date','zone','target','model'];h=k+['timestamp']
    valid=valid_online_forecasts(before)
    plan=valid[valid.source.eq('historical_mock_ensemble')][k].drop_duplicates();plan.run_date=plan.run_date.astype(str)
    # Earlier repaired losses can affect later learned weights; regenerate every
    # historical ensemble date at or after the first corrected component.
    plan=plan[plan.run_date.ge(first)]
    produced=build_selected_ensembles(ForecastStore(scratch),actuals_path=staged/'dashboard/actuals_recent.parquet',run_dates=sorted(plan.run_date.unique()),source='historical_mock_ensemble')
    produced.run_date=produced.run_date.astype(str);incoming=produced.merge(plan,on=k,how='inner',validate='many_to_one')
    if len(incoming[k].drop_duplicates())!=len(plan):raise ValueError('Missing dependent historical ensembles')
    old=before.merge(plan,on=k,how='inner')
    incoming=incoming.drop(columns='actual_mw').merge(old[h+['actual_mw']],on=h,validate='one_to_one')
    q=old[h+['forecast_mw']].merge(incoming[h+['forecast_mw']],on=h,suffixes=('_old','_new'),validate='one_to_one')
    changes=q.assign(changed=~np.isclose(q.forecast_mw_old,q.forecast_mw_new,atol=1e-8,rtol=1e-12)).groupby(k).changed.any()
    for day,part in incoming.groupby('run_date'):store.append_forecasts(part,str(day))
    (store.dashboard_dir()/'leaderboard_daily_status.json').unlink(missing_ok=True)
    after=store.read_forecasts();live=before[before.source.eq('online')][h+['forecast_mw']];check=live.merge(after[after.source.eq('online')][h+['forecast_mw']],on=h,suffixes=('_old','_new'),how='outer',indicator=True,validate='one_to_one')
    assert check._merge.eq('both').all() and np.allclose(check.forecast_mw_old,check.forecast_mw_new,atol=1e-8,rtol=1e-12)
    report=dict(rebuilt_mock_ensemble_groups=len(plan),changed_mock_ensemble_groups=int(changes.sum()),preserved_live_rows=len(live))
    (staged/'ensemble_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged',required=True)
    p.add_argument('--scratch',required=True)
    p.add_argument('--first',required=True)
    a=p.parse_args();rebuild(a.staged,a.scratch,a.first)
