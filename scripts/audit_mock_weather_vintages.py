"""Audit every retrospective group without changing forecasts or running models."""
from pathlib import Path
import argparse,json
import pandas as pd
from rt_forecast_dashboard.data.weather_vintages import WeatherVintages,WeatherVintageError
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp,forecast_horizon_hours

KEYS=['run_date','zone','target','model']

def audit(data,weather,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame=ForecastStore(Path(data)).read_forecasts()
    frame.run_date=frame.run_date.astype(str)
    mock=frame[frame.source.fillna('').str.contains('mock|replay',case=False,regex=True)]
    plan=mock[KEYS+['source']].drop_duplicates()
    vintages=WeatherVintages(weather);reports=[];proofs=[]
    for (day,z,t),group in plan.groupby(KEYS[:3],sort=True):
        cutoff=brussels_cutoff_timestamp(pd.Timestamp(day).date())
        try:
            _,_,proof=vintages.select(z,t,cutoff,2208,forecast_horizon_hours(day))
            status='weather_available';error='';proofs.append(dict(run_date=day,zone=z,target=t,**proof))
        except WeatherVintageError as exc:status='blocked';error=str(exc)
        for row in group.to_dict('records'):reports.append(dict(**row,status=status,error=error))
    result=pd.DataFrame(reports);result.to_csv(output/'mock_weather_audit.csv',index=False)
    (output/'weather_provenance.json').write_text(json.dumps(proofs,indent=2)+'\n')
    summary=dict(groups=len(plan),weather_available_groups=int(result.status.eq('weather_available').sum()),
                 blocked_groups=int(result.status.eq('blocked').sum()),
                 original_rows=len(mock),first=plan.run_date.min(),last=plan.run_date.max(),
                 policy='weather_collection_asof_v1',full_input_certification=False)
    (output/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','weather','output']:p.add_argument('--'+name,required=True)
    audit(**vars(p.parse_args()))
