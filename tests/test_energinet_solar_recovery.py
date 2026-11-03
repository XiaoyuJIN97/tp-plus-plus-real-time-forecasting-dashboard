import pandas as pd
import pytest
from rt_forecast_dashboard.data.energinet_solar_recovery import published_zero_hours,fill_missing_actuals


def samples():
    return pd.DataFrame(dict(Minutes5UTC=pd.date_range('2026-07-29T20:00Z',periods=12,freq='5min'),PriceArea='DK1',SolarPower=0.))


def test_only_twelve_explicit_zero_intervals_recover_an_hour():
    source=samples();result=published_zero_hours(source)
    assert len(result)==1 and result.actual_mw.iloc[0]==0
    assert published_zero_hours(source.iloc[:-1]).empty
    source.loc[0,'SolarPower']=.1
    assert published_zero_hours(source).empty
    source.loc[0,'SolarPower']=float('nan')
    assert published_zero_hours(source).empty


def test_duplicate_intervals_do_not_manufacture_coverage_or_hide_conflicts():
    source=samples()
    assert len(published_zero_hours(pd.concat([source,source])))==1
    assert published_zero_hours(pd.concat([source.iloc[:-1],source.iloc[[0]]])).empty
    duplicate=source.iloc[[0]].copy();duplicate['SolarPower']=1.
    with pytest.raises(ValueError,match='Conflicting'):published_zero_hours(pd.concat([source,duplicate]))


def test_recovery_preserves_existing_actuals_and_zone_and_does_not_extend_range():
    zeros=published_zero_hours(samples())
    stamp=zeros.timestamp_utc.iloc[0]
    for value,count in [(2.,0),(float('nan'),1)]:
        original=pd.DataFrame(dict(timestamp_utc=[stamp],value=[value]))
        result,ledger=fill_missing_actuals(original,zeros,'DK1')
        assert len(ledger)==count
        assert result.value.iloc[0]==(0. if count else 2.)
    original=pd.DataFrame(dict(timestamp_utc=[stamp+pd.Timedelta(hours=1)],value=[1.]))
    result,ledger=fill_missing_actuals(original,zeros,'DK1')
    assert ledger.empty and len(result)==1
    original=pd.DataFrame(dict(timestamp_utc=[stamp],value=[float('nan')]))
    result,ledger=fill_missing_actuals(original,zeros,'DK2')
    assert ledger.empty and result.value.isna().all()
