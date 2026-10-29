"""UTC coverage and unambiguous plotting across both Brussels clock changes."""
import numpy as np
import pandas as pd
import pytest
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp, forecast_horizon_hours
from rt_forecast_dashboard.pipeline.build_selected_ensembles import _complete_run, _band_for_horizon, build_selected_ensembles
from rt_forecast_dashboard.pipeline.check_forecast_complete import is_forecast_complete
from rt_forecast_dashboard.model_onboarding.contracts import frames
from rt_forecast_dashboard.ui.analytics import valid_online_forecasts
from rt_forecast_dashboard.ui.charts import deterministic_forecast_chart, conformal_forecast_chart, forecast_line_chart
from rt_forecast_dashboard.conformal import conformal_intervals
from rt_forecast_dashboard.storage import ForecastStore


def run(day, model='t0_gamma_online'):
    start=brussels_cutoff_timestamp(pd.Timestamp(day).date())
    h=forecast_horizon_hours(day)
    return pd.DataFrame(dict(run_date=day, run_at=start, zone='BE', target='load',
        model=model, model_label=model, source='online',context_hours=2208,
        context_end=start-pd.Timedelta(hours=1),timestamp=pd.date_range(start,periods=h,freq='h'),
        forecast_mw=np.arange(h)+100., actual_mw=np.arange(h)+101.))


@pytest.mark.parametrize('day,hours,next_day',[('2026-10-24',25,'2026-10-25'),
    ('2026-03-28',23,'2026-03-29'),('2026-10-25',24,'2026-10-26')])
def test_cutoff_windows_and_completeness(day,hours,next_day,tmp_path,monkeypatch):
    from rt_forecast_dashboard.pipeline import check_forecast_complete as checker
    from rt_forecast_dashboard.ui import analytics
    from datetime import date
    monkeypatch.setattr(analytics,'latest_complete_run_date',lambda:date(2026,11,1))
    frame=run(day)
    assert len(frame)==hours
    assert frame.timestamp.iloc[-1]+pd.Timedelta(hours=1)==run(next_day).timestamp.iloc[0]
    assert len(_complete_run(frame,['forecast_mw']))==hours
    assert len(valid_online_forecasts(frame))==hours
    context=pd.DataFrame(dict(timestamp=pd.date_range(frame.timestamp.min()-pd.Timedelta(hours=2208),periods=2208,freq='h'),actual_mw=100.))
    assert len(frames(context,frame,[])[1])==hours
    monkeypatch.setattr(checker,'expected_groups',lambda keys:{('BE','load','t0_gamma_online')})
    path=tmp_path/f'forecasts_{day}.csv'
    frame.to_csv(path,index=False)
    assert is_forecast_complete(path)[0]
    frame.iloc[:-1].to_csv(path,index=False)
    assert not is_forecast_complete(path)[0]
    if hours==25:
        assert _band_for_horizon(25,[[1,6],[7,12],[13,18],[19,24]])==3


@pytest.mark.parametrize('chart',[deterministic_forecast_chart,conformal_forecast_chart,forecast_line_chart])
@pytest.mark.parametrize('day',['2026-10-24','2026-03-28'])
def test_plot_uses_unique_monotone_instants_and_explicit_local_labels(chart,day):
    frame=run(day)
    frame['lower_80']=frame.forecast_mw-5
    frame['upper_80']=frame.forecast_mw+5
    # Duplicate delivery rows (e.g. overlapping retained runs) are shown once.
    duplicate=frame.iloc[[10]].copy()
    frame=pd.concat([frame,duplicate],ignore_index=True)
    if chart==conformal_forecast_chart:
        figure=chart(frame,levels=(80,))
    elif chart==deterministic_forecast_chart:
        figure=chart(frame,title='DST')
    else:
        figure=chart(frame)
    for trace in figure.data:
        times=pd.DatetimeIndex(pd.to_datetime(trace.x,utc=True))
        assert times.is_unique and times.is_monotonic_increasing
        assert len(times)==forecast_horizon_hours(day)
    assert figure.to_json()  # Verify timezone-aware coordinates serialize for Plotly.
    labels=list(figure.layout.xaxis.ticktext)
    assert len(labels)==len(set(labels))
    if day=='2026-10-24':
        assert 'Oct 25<br>02:00 CEST<br>→ 02:00 CET' in labels
        hover=[row[-1] for trace in figure.data for row in trace.customdata]
        assert any('02:00 CEST (UTC+0200)' in value for value in hover)
        assert any('02:00 CET (UTC+0100)' in value for value in hover)
    else:
        assert all('Mar 29<br>02:00' not in value for value in labels)


@pytest.mark.parametrize('chart',[deterministic_forecast_chart,conformal_forecast_chart])
def test_multiday_transition_labels_have_room_on_narrow_panels(chart):
    frame=pd.concat([run(str(day.date())) for day in pd.date_range('2026-10-12',periods=14)],ignore_index=True)
    frame['lower_80']=frame.forecast_mw-5
    frame['upper_80']=frame.forecast_mw+5
    figure=chart(frame,levels=(80,)) if chart==conformal_forecast_chart else chart(frame,title='DST')
    axis=figure.layout.xaxis
    ticks=pd.DatetimeIndex(axis.tickvals)
    span=frame.timestamp.max()-frame.timestamp.min()
    # One compact transition label replaces the two adjacent hourly labels.
    assert len(ticks)<=4
    assert (ticks.to_series().diff().dropna()>=span/4).all()
    assert axis.tickangle==0
    assert 'Oct 25<br>02:00 CEST<br>→ 02:00 CET' in axis.ticktext


def test_calibration_accepts_mixed_run_lengths_and_calibrates_extra_hour():
    days=pd.date_range('2026-10-01',periods=30)
    frame=pd.concat([run(str(day.date())) for day in days],ignore_index=True)
    result=conformal_intervals(frame,min_runs=20)
    autumn=result[result.run_date.eq('2026-10-24')]
    assert len(autumn)==25 and autumn.lower_80.notna().all()
    assert result[result.run_date.eq('2026-10-26')].lower_80.notna().all()
    # Future outcomes must never alter the earlier bands.
    frame.loc[frame.run_date.ge('2026-10-24'),'actual_mw']=1e9
    changed=conformal_intervals(frame,min_runs=20)
    pd.testing.assert_series_equal(result.loc[result.run_date.le('2026-10-24'),'lower_80'],
                                   changed.loc[changed.run_date.le('2026-10-24'),'lower_80'])


@pytest.mark.parametrize('day',['2026-10-24','2026-03-28'])
def test_all_ensemble_strategies_keep_dst_hourly_grid(day,tmp_path,monkeypatch):
    from rt_forecast_dashboard.pipeline import build_selected_ensembles as module
    store=ForecastStore(data_dir=tmp_path/'data')
    candidate=run(day)
    reference=run(day,'chronos2_online')
    store.append_forecasts(pd.concat([candidate,reference]),day)
    actuals=candidate[['zone','target','timestamp','actual_mw']].rename(columns={'timestamp':'timestamp_utc'})
    path=tmp_path/'actuals.parquet';actuals.to_parquet(path,index=False)
    original=module.load_yaml
    settings=original('ensemble_registry.yml')['settings']
    for method in ('simple_mean','median','constrained_learned'):
        monkeypatch.setattr(module,'load_yaml',lambda name:dict(settings=settings,selections={'BE':{'load':dict(method=method,include_tso=False)}}) if name=='ensemble_registry.yml' else original(name))
        result=build_selected_ensembles(store,run_dates=[day],actuals_path=path,offline_weights_path=tmp_path/'none.csv')
        assert len(result)==forecast_horizon_hours(day)
        assert result.timestamp.tolist()==candidate.timestamp.tolist()


@pytest.mark.parametrize('day',['2026-10-24','2026-03-28'])
def test_daily_models_and_weather_share_the_dst_horizon(day,tmp_path,monkeypatch):
    from rt_forecast_dashboard.pipeline import run_daily as daily
    from rt_forecast_dashboard.data.demo import demo_future_frame, demo_context_frame
    from rt_forecast_dashboard.models.foundation_online import FoundationOnlineAdapter
    date=pd.Timestamp(day).date()
    hours=forecast_horizon_hours(day)
    calls=[]
    class Power:
        def fetch_tso_forecast(self,**kwargs):
            calls.append(('power',kwargs['horizon_hours']))
            return demo_future_frame(date,'BE','load',kwargs['horizon_hours'])[['timestamp','tso_forecast_mw']]
        def fetch_actual_context(self,**kwargs):
            return demo_context_frame(date,'BE','load',kwargs['context_hours'])[['timestamp','actual_mw','tso_forecast_mw']]
    class Weather:
        def fetch_future_weather(self,**kwargs):
            calls.append(('weather',kwargs['horizon_hours']))
            return demo_future_frame(date,'BE','load',kwargs['horizon_hours']).drop(columns=['actual_mw','tso_forecast_mw'])
        def fetch_context_weather(self,**kwargs):
            return demo_context_frame(date,'BE','load',kwargs['context_hours']).drop(columns=['actual_mw','tso_forecast_mw'])
    def native(self,contexts,covariates,horizon):
        assert horizon==hours and covariates[0].shape[1]==2208+hours
        calls.append((self.model,horizon))
        return np.ones((len(contexts),horizon))
    monkeypatch.setattr(daily,'ForecastStore',lambda:ForecastStore(data_dir=tmp_path/'data'))
    monkeypatch.setattr(daily,'EntsoeForecastClient',Power)
    monkeypatch.setattr(daily,'OpenMeteoClient',Weather)
    monkeypatch.setattr(daily,'zones',lambda:{'BE':{'entsoe_code':'BE','targets':['load']}})
    monkeypatch.setattr(daily,'selected_weather_points',lambda *args:[])
    monkeypatch.setattr(daily,'iter_model_adapters',lambda target:[
        (name+'_online',{'targets':['load']},FoundationOnlineAdapter(name)) for name in ('t0_gamma','timestral')])
    monkeypatch.setattr(FoundationOnlineAdapter,'predict_batch',native)
    result=daily.run_daily_forecast(date)
    assert len(result)==hours*2
    assert result.groupby('model').timestamp.nunique().eq(hours).all()
    assert calls==[('power',hours),('weather',hours),('t0_gamma',hours),('timestral',hours)]
    calls.clear()
    assert daily.run_daily_forecast(date,only_missing=True).empty
    assert calls==[]  # No model inference or data requests for complete groups.
    store=ForecastStore(data_dir=tmp_path/'data')
    saved=pd.read_csv(store.forecast_path(day))
    saved.loc[saved.model.eq('t0_gamma_online'),'forecast_mw']=123.
    saved.loc[saved.model.eq('timestral_online'),'forecast_mw']=np.nan
    saved.to_csv(store.forecast_path(day),index=False)
    recovered=daily.run_daily_forecast(date,only_missing=True)
    assert len(recovered)==hours and set(recovered.model)=={'timestral_online'}
    assert calls==[('power',hours),('weather',hours),('timestral',hours)]
    assert store.read_forecasts().query("model == 't0_gamma_online'").forecast_mw.eq(123.).all()


def test_rolling_learned_weights_handle_mixed_dst_training_lengths(tmp_path,monkeypatch):
    from rt_forecast_dashboard.pipeline import build_selected_ensembles as module
    store=ForecastStore(data_dir=tmp_path/'data')
    outcomes=[]
    days=[str(day.date()) for day in pd.date_range('2026-10-01',periods=26)]
    for day in days:
        first=run(day)
        second=run(day,'chronos2_online');second['forecast_mw']+=2
        store.append_forecasts(pd.concat([first,second]),day)
        outcomes.append(first[['zone','target','timestamp','actual_mw']])
    actuals=pd.concat(outcomes).rename(columns={'timestamp':'timestamp_utc'})
    path=tmp_path/'actuals.parquet';actuals.to_parquet(path,index=False)
    original=module.load_yaml
    settings=original('ensemble_registry.yml')['settings']
    monkeypatch.setattr(module,'load_yaml',lambda name:dict(settings=settings,selections={'BE':{'load':dict(method='constrained_learned',include_tso=False)}}) if name=='ensemble_registry.yml' else original(name))
    result=build_selected_ensembles(store,run_dates=days[-3:],actuals_path=path,offline_weights_path=tmp_path/'none.csv')
    assert result.groupby('run_date').size().to_dict()=={'2026-10-24':25,'2026-10-25':24,'2026-10-26':24}
    assert result.covariate_case.str.startswith('rolling_prior_actuals:').all()


def test_isolated_onboarding_worker_exports_all_transition_hours(tmp_path):
    import json
    import sys
    import yaml
    from pathlib import Path
    from rt_forecast_dashboard.model_onboarding.runner import run as evaluate_run
    from rt_forecast_dashboard.model_onboarding.contracts import records
    root=Path(__file__).resolve().parents[1]
    spec=yaml.safe_load((root/'examples/model_onboarding/manifest.yml').read_text())
    spec['model_key']='dst_example'
    spec['scope']=[dict(zone='BE',target='load',covariates=[])]
    manifest=tmp_path/'manifest.yml';manifest.write_text(yaml.safe_dump(spec))
    windows=tmp_path/'windows.jsonl'
    with windows.open('w') as handle:
        for day in ('2026-03-28','2026-10-24'):
            future=run(day)
            origin=future.timestamp.min()
            context=pd.DataFrame(dict(timestamp=pd.date_range(origin-pd.Timedelta(hours=2208),periods=2208,freq='h'),actual_mw=100.))
            handle.write(json.dumps(dict(zone='BE',target='load',run_date=day,
                origin_utc=origin.isoformat(),inputs_available_at_utc=origin.isoformat(),
                context=records(context),future=records(future[['timestamp']]),
                covariates=[],actuals=future.actual_mw.tolist()))+'\n')
    output=tmp_path/'run'
    assert evaluate_run(manifest,windows,output,sys.executable,root/'config/model_onboarding.yml','offline')
    forecasts=pd.read_csv(output/'forecasts.csv')
    assert forecasts.groupby('run_date').size().to_dict()=={'2026-03-28':23,'2026-10-24':25}
