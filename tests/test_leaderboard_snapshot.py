from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.pipeline.leaderboard_snapshot import build_daily_leaderboard
from rt_forecast_dashboard.storage import ForecastStore


def forecasts() -> pd.DataFrame:
    times = pd.date_range('2026-07-24T16:00:00Z', periods=24, freq='h')
    parts = []
    for model, label, error in [('tso_reference','TSO forecast',10), ('chronos2_online','Chronos2',5)]:
        parts.append(pd.DataFrame({'run_date':'2026-07-24','zone':'BE','target':'load',
                                  'model':model,'model_label':label,'timestamp':times,
                                  'forecast_mw':100+error,'actual_mw':100.,'source':'online',
                                  'context_hours':2208,'context_end':times.min()-pd.Timedelta(hours=1)}))
    return pd.concat(parts,ignore_index=True)


def test_daily_snapshot_is_stable_and_refreshes_next_brussels_day(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path)
    frame = forecasts()
    first = build_daily_leaderboard(store,frame,now=pd.Timestamp('2026-10-15T12:00:00Z'))
    assert first.set_index('display_family').loc['Chronos2','MAE'] == 5
    frame.loc[frame.model.eq('chronos2_online'),'forecast_mw'] = 120
    repeated = build_daily_leaderboard(store,frame,now=pd.Timestamp('2026-10-15T20:00:00Z'))
    pd.testing.assert_frame_equal(first,repeated)
    next_day = build_daily_leaderboard(store,frame,now=pd.Timestamp('2026-10-15T22:01:00Z'))
    assert next_day.snapshot_date.eq('2026-10-16').all()
    assert next_day.set_index('display_family').loc['Chronos2','MAE'] == 20
    assert next_day.set_index('display_family').loc['Chronos2','rMAE'] == 2


def test_incomplete_actuals_do_not_lock_an_empty_snapshot_for_the_day(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path)
    frame = forecasts()
    frame.loc[0,'actual_mw'] = float('nan')
    now = pd.Timestamp('2026-10-15T12:00:00Z')
    assert build_daily_leaderboard(store,frame,now=now).empty
    assert not (store.dashboard_dir()/'leaderboard_daily.parquet').exists()
    assert not build_daily_leaderboard(store,forecasts(),now=now).empty
