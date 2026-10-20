import numpy as np
import pandas as pd
from rt_forecast_dashboard.conformal import conformal_intervals, conformal_radius, interval_diagnostics
from rt_forecast_dashboard.ui.charts import conformal_forecast_chart


def forecasts(days=32, model='persistence', error=2.):
    rows = []
    for index, day in enumerate(pd.date_range('2026-07-24', periods=days)):
        for hour, timestamp in enumerate(pd.date_range(day.tz_localize('UTC')+pd.Timedelta(hours=16), periods=24, freq='h')):
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
    np.testing.assert_array_equal(latest[latest.model.eq('a')].upper_90.to_numpy(), np.repeat([101.,102.,103.,104.],6))
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
    assert metrics['Scored hours'].eq(10*24).all()


def test_plot_band_segments_do_not_bridge_missing_days():
    bands = conformal_intervals(forecasts())
    days = sorted(bands.run_date.unique())
    sample = bands[bands.run_date.isin([days[-3], days[-1]])]
    figure = conformal_forecast_chart(sample, [80,95], 'One model')
    assert sum(trace.fill == 'tonexty' for trace in figure.data) == 4
    for trace in figure.data:
        times = pd.Series(pd.to_datetime(trace.x))
        assert not times.diff().gt(pd.Timedelta(hours=1)).any()
    assert {trace.name for trace in figure.data if trace.showlegend} == {'80% band', '95% band', 'Forecast', 'Actual'}


def test_model_and_coverage_controls(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from rt_forecast_dashboard.ui import layout
    from rt_forecast_dashboard.storage import ForecastStore
    bands = conformal_intervals(pd.concat([forecasts(model='a'), forecasts(model='b')], ignore_index=True))
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
    at.multiselect[0].set_value(['a']).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 1
    at.multiselect[0].set_value([]).run()
    assert len(at.get('plotly_chart')) == 0
    at.multiselect[0].set_value(['a', 'b']).run()
    at.selectbox[0].set_value(1).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 2
    assert len(at.expander) == 0
    import json
    for chart in at.get('plotly_chart'):
        traces = json.loads(chart.proto.spec)['data']
        assert {trace.get('name') for trace in traces if trace.get('fill') == 'tonexty'} == {'80% band'}


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
    assert status['calibrated_rows'] == 11*24*2
