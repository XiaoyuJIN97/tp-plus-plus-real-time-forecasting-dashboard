from __future__ import annotations

import numpy as np
import pandas as pd
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp
import pytest

from rt_forecast_dashboard.ui.analytics import valid_online_forecasts
from rt_forecast_dashboard.pipeline.build_selected_ensembles import _complete_run, build_selected_ensembles
from rt_forecast_dashboard.storage import ForecastStore
from tests.test_selected_ensembles import _component_rows


def _run(source: str = "online", start: str = "2026-07-24T16:00:00Z") -> pd.DataFrame:
    times = pd.date_range(start, periods=30, freq="h")
    return pd.DataFrame({
        "run_date": "2026-07-24", "run_at": "2026-10-15T08:00:00Z",
        "zone": "BE", "target": "load", "model": "timesfm3_online",
        "source": source, "context_hours": 2208,
        "context_end": times.min() - pd.Timedelta(hours=1),
        "timestamp": times, "forecast_mw": np.arange(30, dtype=float),
    })


@pytest.mark.parametrize("source", ["online", "historical_mock_replay", "historical_mock_realized", "historical_mock_ensemble"])
def test_saved_replays_remain_visible_with_provenance(source) -> None:
    result = valid_online_forecasts(_run(source))
    assert len(result) == 30
    assert result.source.eq(source).all()
    assert result.horizon.tolist() == list(range(30))


@pytest.mark.parametrize("source", ["demo", "legacy", "unknown"])
def test_unverified_or_demo_history_is_not_published(source) -> None:
    assert valid_online_forecasts(_run(source)).empty


def test_twenty_four_rows_on_the_wrong_window_are_rejected() -> None:
    assert valid_online_forecasts(_run(start="2026-07-24T18:00:00Z")).empty


def test_irregular_and_overlong_runs_are_rejected() -> None:
    frame = _run()
    frame.loc[23, "timestamp"] += pd.Timedelta(minutes=30)
    assert valid_online_forecasts(frame).empty
    assert _complete_run(frame, ["forecast_mw"]).empty
    frame = _run()
    extra = frame.iloc[[-1]].copy()
    extra["timestamp"] += pd.Timedelta(hours=1)
    assert valid_online_forecasts(pd.concat([frame, extra])).empty


def test_ensemble_can_start_before_twenty_one_training_runs_without_future_data(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path / "data")
    day = "2026-07-24"
    actual = np.linspace(100, 200, 30)
    store.append_forecasts(_component_rows(day, "BE", "load", actual), day)
    path = tmp_path / "actuals.parquet"
    # Current outcomes must have no effect on warmup weights.
    pd.DataFrame({"timestamp_utc": pd.date_range(brussels_cutoff_timestamp(pd.Timestamp(day).date()), periods=30, freq="h"),
                  "zone": "BE", "target": "load", "actual_mw": 1e9}).to_parquet(path)
    result = build_selected_ensembles(store, actuals_path=path, run_dates=[day],
                                      offline_weights_path=tmp_path / "absent.csv",
                                      source="historical_mock_ensemble")
    assert len(result) == 30
    assert result.covariate_case.str.startswith("equal_weight_warmup:").all()
    np.testing.assert_allclose(result.forecast_mw, actual + (10 - 5 + 2) / 3)


def test_context_cannot_extend_into_the_delivery_window() -> None:
    frame = _run()
    frame["context_end"] = frame.timestamp.min()
    assert valid_online_forecasts(frame).empty


def test_nonfinite_forecasts_do_not_count_as_complete_runs() -> None:
    frame = _run()
    frame.loc[5, "forecast_mw"] = np.inf
    assert valid_online_forecasts(frame).empty


def test_replay_import_repairs_wrong_window_without_overwriting_valid_online_run(tmp_path, monkeypatch) -> None:
    from scripts import align_model_history as alignment

    monkeypatch.setattr(alignment, "zones", lambda: {"BE": {"targets": ["load"]}})
    store = ForecastStore(data_dir=tmp_path / "data")
    existing = _run().assign(model="tso_reference", forecast_mw=80.0)
    wrong = _run(start="2026-07-24T18:00:00Z").assign(model="chronos2_online", forecast_mw=90.0)
    store.append_forecasts(pd.concat([existing, wrong]), "2026-07-24")
    replay = pd.concat([_run().assign(model="tso_reference", forecast_mw=100.0),
                        _run().assign(model="chronos2_online", forecast_mw=110.0)])
    path = tmp_path / "replay.parquet"
    replay.to_parquet(path)
    assert alignment.import_replay(store, path, "2026-07-24", "2026-07-24") == 30
    saved = store.read_forecasts()
    assert saved.query("model == 'tso_reference'").forecast_mw.eq(80.0).all()
    fixed = saved.query("model == 'chronos2_online'")
    assert fixed.forecast_mw.eq(110.0).all()
    assert fixed.source.eq("historical_mock_replay").all()
    assert pd.to_datetime(fixed.timestamp, utc=True).min() == pd.Timestamp("2026-07-24T16:00:00Z")
    assert alignment.import_replay(store, path, "2026-07-24", "2026-07-24") == 0


def test_delivery_grid_tracks_brussels_dst_changes(monkeypatch) -> None:
    from datetime import date
    from rt_forecast_dashboard.ui import analytics

    monkeypatch.setattr(analytics, "latest_complete_run_date", lambda: date(2026, 11, 1))
    frame = _run(start="2026-10-26T17:00:00Z").assign(run_date="2026-10-26")
    assert len(valid_online_forecasts(frame)) == 30
    frame.timestamp -= pd.Timedelta(hours=1)
    frame.context_end -= pd.Timedelta(hours=1)
    assert valid_online_forecasts(frame).empty
