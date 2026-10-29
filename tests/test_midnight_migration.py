"""Production/offline window parity and protection against incomplete input tails."""
import numpy as np
import pandas as pd
import pytest
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp,forecast_end_timestamp,forecast_horizon_hours
from rt_forecast_dashboard.data.entsoe_client import _forecast_on_grid
from rt_forecast_dashboard.models.adapters import PersistenceAdapter
from scripts.plan_midnight_offline import plan
from scripts.batch_replay_midnight import prepare_scope


def test_offline_plan_covers_both_years_and_dst_without_old_24_hour_windows(tmp_path):
    result=plan(tmp_path/'plan.csv',selected_scopes={('BE','load')},selected_models={'t0_gamma_online'})
    assert len(result)==731
    assert set(result.hours)=={29,30,31}
    assert result.loc[result.run_date.eq('2024-03-30'),'hours'].item()==29
    assert result.loc[result.run_date.eq('2025-10-25'),'hours'].item()==31
    assert result.loc[result.run_date.eq('2025-12-31'),'end_utc'].item()=='2026-01-01T23:00:00+00:00'


@pytest.mark.parametrize('day',['2026-10-24','2026-03-28','2026-10-28'])
def test_tso_response_is_trimmed_to_exact_midnight_grid(day):
    start=brussels_cutoff_timestamp(pd.Timestamp(day).date());h=forecast_horizon_hours(day)
    frame=pd.DataFrame(dict(timestamp=pd.date_range(start-pd.Timedelta(hours=18),periods=h+24,freq='h'),tso_forecast_mw=100.))
    selected=_forecast_on_grid(frame,start,h)
    assert len(selected)==h and selected.timestamp.min()==start
    assert selected.timestamp.max()+pd.Timedelta(hours=1)==forecast_end_timestamp(day)
    assert _forecast_on_grid(frame[frame.timestamp.ne(selected.timestamp.max())],start,h).empty
    # A full row count with the wrong delivery hours must not pass validation.
    assert _forecast_on_grid(frame.iloc[:h],start,h).empty


def test_daily_persistence_repeats_profile_for_added_evening_hours():
    origin=brussels_cutoff_timestamp(pd.Timestamp('2026-10-28').date())
    past=pd.DataFrame(dict(timestamp=pd.date_range(origin-pd.Timedelta(hours=168),periods=168,freq='h'),actual_mw=np.arange(168)))
    future=pd.DataFrame(dict(timestamp=pd.date_range(origin,periods=30,freq='h'),tso_forecast_mw=0.))
    pred=PersistenceAdapter().predict(future,'solar',past)
    np.testing.assert_array_equal(pred[24:],pred[:6])


def test_batched_preparation_keeps_delivery_truth_out_of_features(tmp_path):
    day='2026-07-24';origin=brussels_cutoff_timestamp(pd.Timestamp(day).date())
    timestamps=pd.date_range(origin-pd.Timedelta(hours=2208),periods=2238,freq='h')
    for variable,values in [('actual_load',np.r_[np.ones(2208)*100,np.ones(30)*1e12]),('forecast_load',np.ones(2238)*110)]:
        path=tmp_path/'entsoe/data/hourly/BE'/variable/'2026.parquet';path.parent.mkdir(parents=True)
        pd.DataFrame(dict(timestamp_utc=timestamps,value=values)).to_parquet(path,index=False)
    weather=tmp_path/'weather/data/raw/BE/load/2026.csv';weather.parent.mkdir(parents=True)
    pd.DataFrame(dict(timestamp_utc=timestamps,temperature_2m_p1=10.)).to_csv(weather,index=False)
    prepared,issues=prepare_scope(tmp_path,'BE','load',[day])
    assert not issues
    past,future,quality,truth=prepared[day]
    assert len(past)==2208 and len(future)==30
    assert past.actual_mw.eq(100).all()
    assert 'actual_mw' not in future
    assert np.all(truth==1e12)
