from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.pipeline.build_dashboard_artifacts import build_dashboard_forecasts
from rt_forecast_dashboard.storage import ForecastStore


def test_build_dashboard_forecasts_keeps_recent_rows_and_attaches_actuals(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path)
    old = pd.DataFrame(
        {
            "run_date": ["2026-06-01"],
            "run_at": ["2026-06-01T18:00:00Z"],
            "zone": ["BE"],
            "target": ["load"],
            "model": ["test"],
            "timestamp": ["2026-06-01T19:00:00Z"],
            "forecast_mw": [90.0],
        }
    )
    recent = old.copy()
    recent["run_date"] = "2026-10-01"
    recent["run_at"] = "2026-10-01T18:00:00Z"
    recent["timestamp"] = "2026-10-01T19:00:00Z"
    store.append_forecasts(old, "2026-06-01")
    store.append_forecasts(recent, "2026-10-01")

    actuals_path = tmp_path / "actuals.parquet"
    pd.DataFrame(
        {
            "timestamp_utc": ["2026-10-01T19:00:00Z"],
            "collection_time_utc": ["2026-10-01T20:00:00Z"],
            "zone": ["BE"],
            "target": ["load"],
            "actual_mw": [100.0],
        }
    ).to_parquet(actuals_path, index=False)

    path, metadata = build_dashboard_forecasts(store, retention_days=90, actuals_path=actuals_path)
    result = pd.read_parquet(path)

    assert result["run_date"].tolist() == ["2026-10-01"]
    assert result["actual_mw"].tolist() == [100.0]
    assert metadata["rows"] == 1
    assert store.read_dashboard_forecasts()["timestamp"].dt.tz is not None


def test_dashboard_forecasts_falls_back_to_daily_csvs(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path)
    frame = pd.DataFrame(
        {
            "run_date": ["2026-10-01"],
            "run_at": ["2026-10-01T18:00:00Z"],
            "zone": ["BE"],
            "target": ["load"],
            "model": ["test"],
            "timestamp": ["2026-10-01T19:00:00Z"],
            "forecast_mw": [90.0],
        }
    )
    store.append_forecasts(frame, "2026-10-01")

    assert len(store.read_dashboard_forecasts()) == 1
