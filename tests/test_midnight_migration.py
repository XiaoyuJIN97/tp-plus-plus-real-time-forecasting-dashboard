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
    nullable = selected.copy()
    nullable['tso_forecast_mw'] = nullable.tso_forecast_mw.astype('Float64')
    nullable.loc[nullable.index[-1], 'tso_forecast_mw'] = pd.NA
    assert _forecast_on_grid(nullable,start,h).empty


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


def test_assembler_preserves_legacy_groups_but_reports_them_unmigrated(tmp_path):
    from scripts.assemble_midnight_history import assemble
    from rt_forecast_dashboard.storage import ForecastStore
    from rt_forecast_dashboard.ui.analytics import valid_online_forecasts
    day='2026-07-24';origin=brussels_cutoff_timestamp(pd.Timestamp(day).date())
    source=ForecastStore(tmp_path/'source');staged=ForecastStore(tmp_path/'staged')
    def rows(model,hours,tag):
        return pd.DataFrame(dict(run_date=day,zone='BE',target='load',model=model,model_label=model,
            timestamp=pd.date_range(origin,periods=hours,freq='h'),run_at=origin,source=tag,
            forecast_mw=100.,actual_mw=101.,context_hours=2208,context_end=origin-pd.Timedelta(hours=1),
            forecast_window='next_day_midnight_v1' if hours==30 else 'legacy'))
    source.append_forecasts(pd.concat([rows('t0_gamma_online',24,'online'),rows('timestral_online',24,'online')]).assign(context_end='2026-07-24T15:00:00Z'),day)
    staged.append_forecasts(rows('t0_gamma_online',30,'historical_mock_replay'),day)
    coverage=assemble(source.data_dir,[staged.data_dir],tmp_path/'assembled',day,day)
    assert coverage.migrated.sum()==1 and len(coverage)==2
    saved=ForecastStore(tmp_path/'assembled').read_forecasts()
    assert len(saved[saved.model.eq('timestral_online')])==24
    valid=valid_online_forecasts(saved)
    assert len(valid)==30 and set(valid.model)=={'t0_gamma_online'}
    bad=rows('t0_gamma_online',30,'historical_mock_replay');bad.timestamp+=pd.Timedelta(hours=1)
    staged.append_forecasts(bad,day)
    with pytest.raises(ValueError,match='Invalid replay group'):
        assemble(source.data_dir,[staged.data_dir],tmp_path/'rejected',day,day)


def test_interrupted_forecast_write_preserves_previous_complete_file(tmp_path,monkeypatch):
    from rt_forecast_dashboard.storage import ForecastStore
    store=ForecastStore(tmp_path/'data');day='2026-07-24'
    original=pd.DataFrame(dict(run_date=[day],zone=['BE'],target=['load'],model=['t0_gamma_online'],
                               timestamp=['2026-07-24T16:00Z'],run_at=['2026-07-24T16:30Z'],forecast_mw=[100.]))
    store.append_forecasts(original,day)
    before=store.forecast_path(day).read_bytes()
    def interrupted(self,path,**kwargs):
        from pathlib import Path
        Path(path).write_text('incomplete write')
        raise OSError('Interrupted output')
    monkeypatch.setattr(pd.DataFrame,'to_csv',interrupted)
    with pytest.raises(OSError,match='Interrupted'):
        store.append_forecasts(original.assign(forecast_mw=200.),day)
    assert store.forecast_path(day).read_bytes()==before


def test_missing_scope_inputs_are_reported_without_crashing_other_replay_scopes(tmp_path):
    prepared,issues=prepare_scope(tmp_path,'BE','load',['2026-07-24','2026-07-25'])
    assert not prepared and len(issues)==2
    assert all(row['stage']=='prepare' for row in issues)


def test_weather_failure_is_requested_once_and_preserves_weather_independent_models(tmp_path,monkeypatch):
    from rt_forecast_dashboard.pipeline import run_daily as daily
    from rt_forecast_dashboard.data.demo import demo_context_frame,demo_future_frame
    from rt_forecast_dashboard.models.adapters import TsoReferenceAdapter
    from rt_forecast_dashboard.storage import ForecastStore
    from datetime import date
    day=date(2026,7,24);calls=[]
    class Power:
        def fetch_tso_forecast(self,**kwargs):
            return demo_future_frame(day,'BE','load',kwargs['horizon_hours'])[['timestamp','tso_forecast_mw']]
        def fetch_actual_context(self,**kwargs):
            return demo_context_frame(day,'BE','load',kwargs['context_hours'])[['timestamp','actual_mw','tso_forecast_mw']]
    class BrokenWeather:
        def fetch_future_weather(self,**kwargs):
            calls.append('weather');raise RuntimeError('Incomplete forecast weather')
    monkeypatch.setattr(daily,'EntsoeForecastClient',Power)
    monkeypatch.setattr(daily,'OpenMeteoClient',BrokenWeather)
    monkeypatch.setattr(daily,'ForecastStore',lambda:ForecastStore(tmp_path/'data'))
    monkeypatch.setattr(daily,'zones',lambda:{'BE':{'entsoe_code':'BE','targets':['load']}})
    monkeypatch.setattr(daily,'selected_weather_points',lambda *args:[])
    monkeypatch.setattr(daily,'covariates_for',lambda t,z,k:('tso',[]) if k=='tso_reference' else ('weather',['missing_weather']))
    monkeypatch.setattr(daily,'iter_model_adapters',lambda t:[(k,{'targets':['load']},TsoReferenceAdapter())
        for k in ['tso_reference','t0_gamma_online','timestral_online']])
    result=daily.run_daily_forecast(day)
    assert set(result.model)=={'tso_reference'} and len(result)==30
    assert calls==['weather']


def test_overlapping_plot_hours_prefer_latest_origin_even_if_old_run_was_replayed_later():
    from rt_forecast_dashboard.ui.charts import deterministic_forecast_chart
    parts=[]
    for day,value,generated in [('2026-10-24',1.,'2026-11-02'),('2026-10-25',2.,'2026-11-01')]:
        origin=brussels_cutoff_timestamp(pd.Timestamp(day).date())
        parts.append(pd.DataFrame(dict(run_date=day,run_at=generated,zone='BE',target='load',
            model='t0_gamma_online',model_label='T0 Gamma',forecast_mw=value,
            timestamp=pd.date_range(origin,periods=forecast_horizon_hours(day),freq='h'))))
    figure=deterministic_forecast_chart(pd.concat(parts,ignore_index=True),'DST overlap')
    trace=next(t for t in figure.data if t.name=='T0 Gamma')
    times=pd.DatetimeIndex(pd.to_datetime(trace.x,utc=True))
    assert times.is_unique and times.is_monotonic_increasing and len(times)==55
    newer=brussels_cutoff_timestamp(pd.Timestamp('2026-10-25').date())
    assert np.all(np.asarray(trace.y)[times>=newer]==2.)


def test_replay_comparison_rejects_old_24_hour_or_misaligned_baselines():
    from scripts.report_foundation_online import matched_complete_windows
    day='2026-07-24';origin=brussels_cutoff_timestamp(pd.Timestamp(day).date())
    frame=pd.DataFrame(dict(run_date=day,zone='BE',target='load',
        timestamp=pd.date_range(origin,periods=30,freq='h'),context_end=origin-pd.Timedelta(hours=1),
        context_end_baseline=origin-pd.Timedelta(hours=1)))
    assert len(matched_complete_windows(frame))==30
    assert matched_complete_windows(frame.iloc[:24]).empty
    assert matched_complete_windows(frame.assign(context_end_baseline=origin)).empty


def test_missing_past_tso_hour_does_not_discard_observed_actual_context(monkeypatch):
    from rt_forecast_dashboard.data.entsoe_client import EntsoeForecastClient
    from datetime import date
    origin=brussels_cutoff_timestamp(date(2026,7,24))
    actual=pd.DataFrame(dict(timestamp=pd.date_range(origin-pd.Timedelta(hours=4),periods=4,freq='h'),actual_mw=[100.,101.,102.,103.]))
    tso=actual.rename(columns={'actual_mw':'tso_forecast_mw'}).drop(index=2)
    class Archive:
        def fetch_actuals(self,**kwargs):return actual
        def fetch_forecast(self,**kwargs):return tso
    client=EntsoeForecastClient(api_key=None,demo_mode=False);client.archive=Archive()
    result=client.fetch_actual_context(run_date=date(2026,7,24),zone='BE',zone_code='BE',target='load',context_hours=3)
    assert result.actual_mw.tolist()==[101.,102.,103.]
    assert result.loc[result.timestamp.eq(origin-pd.Timedelta(hours=2)),'tso_forecast_mw'].isna().all()
