import numpy as np
import pandas as pd
from rt_forecast_dashboard.conformal import conformal_intervals, conformal_radius, interval_diagnostics
from rt_forecast_dashboard.ui.charts import conformal_forecast_chart
from rt_forecast_dashboard.conformal import interval_leaderboard


def forecasts(days=32, model='persistence', error=2.):
    rows = []
    for index, day in enumerate(pd.date_range('2026-07-24', periods=days)):
        for hour, timestamp in enumerate(pd.date_range(day.tz_localize('UTC')+pd.Timedelta(hours=16), periods=30, freq='h')):
            rows.append(dict(run_date=str(day.date()), zone='BE', target='load', model=model,
                             model_label=model, timestamp=timestamp, forecast_mw=100.,
                             actual_mw=100.+(error(index, hour) if callable(error) else error)))
    return pd.DataFrame(rows)


def test_finite_sample_order_statistic():
    assert conformal_radius(np.arange(1, 21), .9) == 19
    assert np.isinf(conformal_radius([1, 2, 3], .95))


def test_no_current_future_or_unfinished_run_leakage():
    original = forecasts()
    bands = conformal_intervals(original)
    days = sorted(original.run_date.unique())
    assert bands[bands.run_date.eq(days[20])].lower_90.isna().all()
    ready = bands[bands.run_date.eq(days[21])]
    assert ready.calibration_runs.eq(20).all()
    assert ready.lower_90.eq(98.).all()
    assert ready.calibration_last_delivery.max()+pd.Timedelta(hours=3) <= ready.timestamp.min()
    altered = original.copy()
    altered.loc[altered.run_date.ge(days[25]), 'actual_mw'] = 1e9
    recalculated = conformal_intervals(altered)
    pd.testing.assert_frame_equal(bands.loc[bands.run_date.le(days[25]), ['lower_90','upper_90']],
                                  recalculated.loc[recalculated.run_date.le(days[25]), ['lower_90','upper_90']])
    altered = original.copy()
    altered.loc[0, 'actual_mw'] = np.nan
    missing = conformal_intervals(altered)
    assert missing[missing.run_date.eq(days[21])].lower_90.isna().all()


def test_models_and_horizon_buckets_calibrate_separately():
    a = forecasts(model='a', error=lambda day, h: h//6+1)
    b = forecasts(model='b', error=50.)
    result = conformal_intervals(pd.concat([a,b], ignore_index=True))
    latest = result[result.run_date.eq(result.run_date.max())]
    np.testing.assert_array_equal(latest[latest.model.eq('a')].upper_90.to_numpy(), np.repeat([101.,102.,103.,104.,105.],6))
    assert latest[latest.model.eq('b')].upper_90.eq(150.).all()


def test_rolling_window_clipping_and_nested_bands():
    frame = forecasts(error=lambda day, h: day+1.)
    result = conformal_intervals(frame, window_runs=3, min_runs=3)
    sample = result[result.run_date.eq(sorted(result.run_date.unique())[10])]
    assert sample.upper_90.eq(109.).all()  # Only runs 6, 7, 8 are eligible.
    assert sample.lower_95.isna().all()  # 18 scores cannot support the corrected 95% rank.
    nested = conformal_intervals(frame).dropna(subset=["lower_95"])
    assert (nested.lower_95 <= nested.lower_90).all()
    assert (nested.lower_90 <= nested.lower_80).all()
    assert (nested.upper_95 >= nested.upper_90).all()
    assert (nested.upper_90 >= nested.upper_80).all()
    frame.forecast_mw = 1.
    clipped = conformal_intervals(frame)
    assert clipped.lower_95.dropna().ge(0).all()


def test_coverage_scores_only_realized_banded_hours():
    result = conformal_intervals(forecasts())
    last = result.run_date.max()
    result.loc[result.run_date.eq(last), 'actual_mw'] = np.nan
    metrics = interval_diagnostics(result)
    assert metrics['Observed coverage (%)'].eq(100.).all()
    assert metrics['Scored hours'].eq(10*30).all()


def test_interval_leaderboard_matches_complete_observed_calibrated_runs():
    a=forecasts(days=5,model='a',error=lambda day,h:100. if h<6 else 0.)
    b=forecasts(days=5,model='b',error=lambda day,h:100. if h<6 else 0.)
    a['lower_80']=90.;a['upper_80']=110.
    b['lower_80']=80.;b['upper_80']=120.
    # One missing interval excludes the entire run for both models.
    a.loc[23,'upper_80']=np.nan
    # Unobserved future outcomes cannot contribute.
    a.loc[a.run_date.eq(a.run_date.max()),'actual_mw']=np.nan
    b.loc[b.run_date.eq(b.run_date.max()),'actual_mw']=np.nan
    result=interval_leaderboard(pd.concat([a,b],ignore_index=True)).set_index('model')
    assert result.coverage_pct.eq(80.).all()
    assert result.scored_runs.eq(3).all() and result.scored_hours.eq(90).all()
    assert result.loc['a','mean_interval_width_mw']==20.
    assert result.loc['b','mean_interval_width_mw']==40.
    np.testing.assert_allclose(result.winkler_score_mw, 200.)
    # A mismatched realized truth excludes that whole run from both models.
    b.loc[b.run_date.eq(sorted(b.run_date.unique())[1]),'actual_mw']+=1.
    result=interval_leaderboard(pd.concat([a,b],ignore_index=True))
    assert result.scored_runs.eq(2).all()


def test_interval_leaderboard_preserves_dst_windows():
    from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp
    parts=[]
    for model in ('tso_reference','ensemble_selected'):
        for day,hours in (('2026-03-28',29),('2026-10-24',31)):
            part=pd.DataFrame(dict(zone='BE',target='load',model=model,model_label=model,
                run_date=day,actual_mw=100.,timestamp=pd.date_range(
                    brussels_cutoff_timestamp(pd.Timestamp(day).date()),periods=hours,freq='h')))
            part['lower_80']=0.;part['upper_80']=200.
            parts.append(part)
    result=interval_leaderboard(pd.concat(parts,ignore_index=True))
    assert result.scored_hours.eq(60).all() and result.scored_runs.eq(2).all()
    assert result.coverage_pct.eq(100.).all()


def test_probabilistic_page_uses_identical_model_by_zone_tables(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from rt_forecast_dashboard.ui import layout
    from rt_forecast_dashboard.storage import ForecastStore
    scores = pd.DataFrame([
        dict(zone=z,target='load',model=m,model_label=label,coverage_pct=coverage,
             mean_interval_width_mw=width,winkler_score_mw=winkler,scored_runs=30,scored_hours=900)
        for z,coverage,width,winkler in [('BE',81.,200.,250.),('DE',79.,400.,500.)]
        for m,label in [('ensemble_selected','Ensembled model'),('chronos2_online','Chronos2 with covariates'),('new_model','New foundation model')]
    ] + [dict(zone='NL',target='load',model='hidden',model_label='Hidden area',coverage_pct=80.,mean_interval_width_mw=1.,winkler_score_mw=2.,scored_runs=30,scored_hours=900)])
    monkeypatch.setattr(layout,'_probabilistic_leaderboard',lambda version,target:scores[scores.target.eq(target)].copy())
    monkeypatch.setattr(layout,'ForecastStore',lambda:ForecastStore(data_dir=tmp_path))
    monkeypatch.setattr(layout,'_leaderboard_summary',lambda *args:pd.DataFrame())
    def app():
        from rt_forecast_dashboard.ui.layout import _render_probabilistic_forecasts_page
        _render_probabilistic_forecasts_page()
    at=AppTest.from_function(app).run()
    at.selectbox[0].select('load').run()
    assert not at.exception
    assert len(at.dataframe)==3
    for table in at.dataframe:
        assert list(table.value)==['Model Family','BE','DE']
        assert table.value['Model Family'].tolist()==['Ensembled model','Chronos2','New foundation model']
    assert at.dataframe[0].value.loc[0,'BE']==250.
    assert at.dataframe[1].value.loc[0,'BE']==200.
    assert at.dataframe[2].value.loc[0,'BE']==81.
    headings=[m.value for m in at.markdown]
    assert '#### Winkler score (MW)' in headings
    assert '#### Mean interval width (MW)' in headings
    assert '#### Coverage (%)' in headings


def test_point_leaderboard_does_not_load_probability_scores(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from rt_forecast_dashboard.ui import layout
    from rt_forecast_dashboard.storage import ForecastStore
    store=ForecastStore(data_dir=tmp_path)
    directory=store.dashboard_dir();directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([dict(zone='BE',target='load',display_family=m,rMAE=1.,MAE=100.,win_rate=50.)
                  for m in ['Ensembled model','TSO forecast']]).to_parquet(directory/'leaderboard_daily.parquet')
    monkeypatch.setattr(layout,'ForecastStore',lambda:store)
    def forbidden(*args):
        raise AssertionError('Point page must not load interval scores')
    monkeypatch.setattr(layout,'_render_probabilistic_leaderboard',forbidden)
    def app():
        import pandas as pd
        from rt_forecast_dashboard.ui.layout import _render_rmae_landing
        _render_rmae_landing(pd.DataFrame(dict(target=['load'],timestamp=pd.to_datetime(['2026-11-01T00:00Z']),actual_mw=[100.])))
    at=AppTest.from_function(app).run()
    at.selectbox[0].select('load').run()
    assert not at.exception and len(at.dataframe)==3


def test_plot_band_segments_do_not_bridge_missing_days():
    bands = conformal_intervals(forecasts())
    days = sorted(bands.run_date.unique())
    sample = bands[bands.run_date.isin([days[-3], days[-1]])]
    figure = conformal_forecast_chart(sample, [80,95], 'One model')
    assert sum(trace.fill == 'tonexty' for trace in figure.data) == 4
    for trace in figure.data:
        times = pd.Series(pd.to_datetime(trace.x))
        assert not times.diff().gt(pd.Timedelta(hours=1)).any()
    assert {trace.name for trace in figure.data if trace.showlegend} == {'80% interval', '95% interval', 'Forecast', 'Actual'}


def test_model_and_coverage_controls(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from rt_forecast_dashboard.ui import layout
    from rt_forecast_dashboard.storage import ForecastStore
    bands = conformal_intervals(pd.concat([forecasts(model='ensemble_selected'), forecasts(model='tso_reference'), forecasts(model='a')], ignore_index=True))
    monkeypatch.setattr(layout, '_conformal_series', lambda *args: bands)
    monkeypatch.setattr(layout, 'ForecastStore', lambda: ForecastStore(data_dir=tmp_path))

    def app():
        import streamlit as st
        from rt_forecast_dashboard.ui.layout import _render_conformal_section
        _render_conformal_section('load', 'BE', st.session_state['current'])

    at = AppTest.from_function(app)
    at.session_state['current'] = bands
    at.run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 2
    assert set(at.multiselect[0].value) == {'ensemble_selected','tso_reference'}
    assert 'a' in at.multiselect[0].options
    at.multiselect[0].set_value(['a']).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 1
    at.multiselect[0].set_value([]).run()
    assert len(at.get('plotly_chart')) == 0
    at.multiselect[0].set_value(['a', 'tso_reference']).run()
    at.selectbox[0].set_value(1).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 2
    assert len(at.expander) == 0
    import json
    for chart in at.get('plotly_chart'):
        traces = json.loads(chart.proto.spec)['data']
        assert {trace.get('name') for trace in traces if trace.get('fill') == 'tonexty'} == {'80% interval'}


def test_artifact_contains_every_valid_model_and_preserves_warmup(tmp_path):
    from rt_forecast_dashboard.pipeline.build_conformal_artifacts import build_conformal_artifacts
    from rt_forecast_dashboard.storage import ForecastStore
    frame = pd.concat([forecasts(), forecasts(model='timesfm3_online')], ignore_index=True)
    frame['context_hours'] = 2208
    frame['context_end'] = frame.groupby(['run_date', 'model']).timestamp.transform('min')-pd.Timedelta(hours=1)
    frame['source'] = 'online'
    path, status = build_conformal_artifacts(ForecastStore(data_dir=tmp_path), frame)
    result = pd.read_parquet(path)
    assert set(result.model) == set(frame.model)
    assert len(result) == len(frame)
    assert result[result.run_date.eq(result.run_date.min())].lower_90.isna().all()
    assert result[result.run_date.eq(result.run_date.max())].lower_90.notna().all()
    assert status['calibrated_rows'] == 11*30*2
    snapshot=pd.read_parquet(path.parent/'probabilistic_leaderboard.parquet')
    pd.testing.assert_frame_equal(snapshot,interval_leaderboard(result))


def test_scoped_page_load_and_snapshot_do_not_recalculate_or_fetch(tmp_path,monkeypatch):
    from rt_forecast_dashboard.ui import layout
    from rt_forecast_dashboard.storage import ForecastStore
    store=ForecastStore(data_dir=tmp_path)
    frame=forecasts()
    frame['source']='online';frame['context_hours']=2208
    frame['context_end']=frame.groupby('run_date').timestamp.transform('min')-pd.Timedelta(hours=1)
    extra=frame.copy();extra['zone']='FR';extra['target']='solar'
    store.write_dashboard_forecasts(pd.concat([frame,extra],ignore_index=True))
    scores=pd.DataFrame([dict(zone='BE',target='load',model='persistence',model_label='Persistence',coverage_pct=80.,mean_interval_width_mw=4.,scored_runs=3,scored_hours=72)])
    scores.to_parquet(store.dashboard_dir()/'probabilistic_leaderboard.parquet',index=False)
    monkeypatch.setattr(layout,'ForecastStore',lambda:store)
    def forbidden(*args,**kwargs):
        raise AssertionError('Page must not prepare full history or recalculate published scores')
    monkeypatch.setattr(layout,'_prepared_online_forecasts',forbidden)
    monkeypatch.setattr(layout,'interval_leaderboard',forbidden)
    layout._prepared_forecast_series.clear();layout._probabilistic_leaderboard.clear()
    selected=layout._prepared_forecast_series((1234,1),'load','BE')
    assert len(selected)==len(frame) and set(selected.zone)=={'BE'} and set(selected.target)=={'load'}
    pd.testing.assert_frame_equal(layout._probabilistic_leaderboard((1234,1),'load'),scores)
    layout._prepared_forecast_series.clear();layout._probabilistic_leaderboard.clear()


def test_only_selected_forecast_view_is_built(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from rt_forecast_dashboard.ui import layout
    called=[]
    monkeypatch.setattr(layout,'_render_conformal_section',lambda *args:called.append('probabilistic'))
    def app():
        import streamlit as st
        from rt_forecast_dashboard.ui.layout import _render_target_section
        _render_target_section('load',st.session_state['forecasts'],['BE'])
    at=AppTest.from_function(app)
    at.session_state['forecasts']=forecasts()
    at.run()
    assert not at.exception and called==[]
    assert len(at.get('plotly_chart'))==1
    at.radio[0].set_value('Probabilistic forecasts').run()
    assert not at.exception and called==['probabilistic']
    assert len(at.get('plotly_chart'))==0



def test_winkler_score_penalizes_both_tails_and_keeps_boundary_values_inside():
    from rt_forecast_dashboard.conformal import winkler_score
    import pytest
    actual=np.array([100.,85.,115.,90.,110.])
    np.testing.assert_allclose(winkler_score(actual,90.,110.,.8),[20.,70.,70.,20.,20.])
    np.testing.assert_allclose(winkler_score(actual,90.,110.,.9),[20.,120.,120.,20.,20.])
    for level in [0,1,-.1,float('nan')]:
        with pytest.raises(ValueError):winkler_score([100.],[90.],[110.],level)
    with pytest.raises(ValueError):winkler_score([100.],[110.],[90.])
    with pytest.raises(ValueError):winkler_score([float('nan')],[90.],[110.])


def test_probability_page_handles_no_scores(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from rt_forecast_dashboard.ui import layout
    from rt_forecast_dashboard.storage import ForecastStore
    monkeypatch.setattr(layout,'_probabilistic_leaderboard',lambda *args:pd.DataFrame())
    monkeypatch.setattr(layout,'ForecastStore',lambda:ForecastStore(data_dir=tmp_path))
    monkeypatch.setattr(layout,'_leaderboard_summary',lambda *args:pd.DataFrame())
    def app():
        from rt_forecast_dashboard.ui.layout import _render_probabilistic_forecasts_page
        _render_probabilistic_forecasts_page()
    at=AppTest.from_function(app).run()
    assert not at.exception and len(at.dataframe)==0
    assert len(at.info)==1
