"""Create complete 2024/2025 daily evaluation plans for the production window.

Run with the shared replay executor and normalized historical collector inputs.
No reuse of old 24-hour reference forecasts. This plan does not attest that
reconstructed historical input revisions were available at forecast issue time.
"""
import argparse
from pathlib import Path
import pandas as pd
from rt_forecast_dashboard.config import zones,model_registry,model_supports_scope
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp,forecast_end_timestamp,forecast_horizon_hours


def plan(output, years=(2024,2025), selected_scopes=None, selected_models=None):
    rows=[]
    for year in years:
        for day in pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D'):
            for zone,spec in zones().items():
                for target in spec['targets']:
                    if selected_scopes and (zone,target) not in selected_scopes:
                        continue
                    for model,cfg in model_registry().items():
                        if (not cfg.get('enabled') or cfg.get('pipeline_generated') or model=='tabpfn_online'
                                or not model_supports_scope(cfg,zone,target)
                                or target not in cfg.get('targets',[]) or (selected_models and model not in selected_models)):
                            continue
                        rows.append(dict(run_date=str(day.date()),zone=zone,target=target,model=model,
                            origin_utc=brussels_cutoff_timestamp(day.date()).isoformat(),
                            end_utc=forecast_end_timestamp(day.date()).isoformat(),hours=forecast_horizon_hours(day)))
    frame=pd.DataFrame(rows)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False)
    return frame


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--years',default='2024,2025')
    p.add_argument('--scopes',help='Comma-separated zone:target pairs')
    p.add_argument('--models',help='Comma-separated registered model keys')
    a=p.parse_args()
    f=plan(a.output,[int(y) for y in a.years.split(',')],
           {tuple(s.split(':')) for s in a.scopes.split(',')} if a.scopes else None,
           set(a.models.split(',')) if a.models else None)
    print(len(f),'model windows; nominal 30-hour horizons; DST handled by UTC endpoints')
