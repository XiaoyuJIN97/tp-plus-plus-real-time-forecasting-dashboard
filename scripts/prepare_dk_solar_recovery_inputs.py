"""Stage DK solar replay inputs using explicit Energinet zero-hour evidence.

Reads frozen published observations; performs no API requests. Rebuilds only
missing archived actuals, never forecasts or already-published observations.
"""
from pathlib import Path
import argparse
import shutil
import pandas as pd
from rt_forecast_dashboard.data.energinet_solar_recovery import published_zero_hours,fill_missing_actuals
from rt_forecast_dashboard.model_onboarding.contracts import staging_path


def prepare(source,observations,output):
    source=Path(source).resolve();output=staging_path(output)
    if output.exists() or output.is_relative_to(source):raise ValueError('Use a new isolated output directory')
    zeros=published_zero_hours(pd.read_parquet(observations));ledgers=[]
    for zone in ['DK1','DK2']:
        for variable in ['actual_solar_generation','forecast_solar_generation']:
            relative=Path('entsoe/data/hourly')/zone/variable
            shutil.copytree(source/relative,output/relative)
        relative=Path('weather/data/raw')/zone/'solar'
        shutil.copytree(source/relative,output/relative)
        path=output/'entsoe/data/hourly'/zone/'actual_solar_generation/2026.parquet'
        merged,ledger=fill_missing_actuals(pd.read_parquet(path),zeros,zone)
        merged.to_parquet(path,index=False);ledgers.append(ledger)
    pd.concat(ledgers,ignore_index=True).to_parquet(output.parent/'recovered_published_zero_actuals.parquet',index=False)
    return pd.concat(ledgers,ignore_index=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','observations','output']:p.add_argument('--'+key,required=True)
    print(prepare(**vars(p.parse_args())).groupby('zone').size().to_string())
