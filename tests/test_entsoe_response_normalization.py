import numpy as np
import pandas as pd
import pytest

from rt_forecast_dashboard.data.entsoe_client import _entsoe_response_to_hourly_frame, _query_entsoe_hourly


def monthly_response(start, end):
    index = pd.date_range(start, end, freq='h', inclusive='left')
    if start.month == 7:
        return pd.DataFrame({('Wind Offshore', 'Actual Aggregated'): 100.,
                             ('Wind Offshore', 'Actual Consumption'): 7.}, index=index)
    return pd.DataFrame({'Wind Offshore': 200.}, index=index)


def test_normalizes_mixed_months_before_upstream_concatenation():
    from entsoe.decorators import month_limited
    calls = []
    @month_limited
    def query(*, start, end, country_code, psr_type):
        calls.append((start, end))
        return monthly_response(start, end)
    start, end = pd.Timestamp('2026-07-15T16:00Z'), pd.Timestamp('2026-09-10T16:00Z')
    result = _query_entsoe_hourly(query, 'actual_mw', start=start, end=end,
                                  country_code='BE', psr_type='B18')
    assert len(calls) == 2
    assert list(result.columns) == ['timestamp', 'actual_mw']
    assert len(result) == int((end-start)/pd.Timedelta(hours=1))
    assert not result.timestamp.duplicated().any()
    assert result.actual_mw.iloc[0] == 100.  # Consumption must not be added.
    assert result.actual_mw.iloc[-1] == 200.
    assert result.timestamp.max() < end


def test_production_pandas_reproduces_original_error():
    if int(pd.__version__.split('.')[0]) < 3:
        pytest.skip('Original concatenation error occurs on production pandas 3')
    start, split, end = map(pd.Timestamp, ['2026-07-15T16:00Z', '2026-08-15T16:00Z', '2026-09-10T16:00Z'])
    with pytest.raises(NotImplementedError, match='Can only union MultiIndex'):
        pd.concat([monthly_response(start, split), monthly_response(split, end)])


def test_preserves_missing_generation_and_averages_subhourly_values():
    index = pd.date_range('2026-10-17T00:00Z', periods=8, freq='15min')
    values = pd.DataFrame({('Wind Offshore', 'Actual Aggregated'): [np.nan]*4 + [10.,20.,30.,40.],
                           ('Wind Offshore', 'Actual Consumption'): [7.]*8}, index=index)
    result = _entsoe_response_to_hourly_frame(values, 'actual_mw')
    assert len(result) == 1
    assert result.actual_mw.iloc[0] == 25.
    assert result.timestamp.iloc[0] == index[4]


def test_missing_month_is_left_missing():
    from entsoe.exceptions import NoMatchingDataError
    def query(*, start, end):
        if start.month == 8:
            raise NoMatchingDataError
        return monthly_response(start, end)
    result = _query_entsoe_hourly(query, 'actual_mw', start=pd.Timestamp('2026-07-01T00:00Z'), end=pd.Timestamp('2026-10-01T00:00Z'))
    assert not result.timestamp.dt.month.eq(8).any()
    assert len(result) == (31+30)*24
