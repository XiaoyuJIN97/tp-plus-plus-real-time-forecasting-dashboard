import pandas as pd
from rt_forecast_dashboard.ui.layout import _coverage_endpoints, _analysis_window_sentence


def test_coverage_summary_preserves_task_dates_and_latest_forecast():
    frame = pd.DataFrame({
        'target': ['load', 'load', 'solar', 'solar', 'load'],
        'timestamp': pd.to_datetime(['2026-07-24T16:00Z', '2026-10-20T07:00Z',
                                     '2026-07-25T16:00Z', '2026-10-20T06:00Z', '2026-10-20T15:00Z']),
        'actual_mw': [100., 200., 5., 6., float('nan')],
        'run_date': ['2026-07-24', '2026-10-19', '2026-07-25', '2026-10-19', '2026-10-19'],
        'source': ['historical_mock_replay', 'online', 'historical_mock_replay', 'online', 'online'],
    })
    summary = _coverage_endpoints(frame)
    for target in ['load', 'solar']:
        assert _analysis_window_sentence(summary[summary.target.eq(target)]) == _analysis_window_sentence(frame[frame.target.eq(target)])
    assert summary.run_date.max() == frame.run_date.max()
    assert summary.source.str.startswith('historical_mock').any()
