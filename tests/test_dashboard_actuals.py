from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.data import dashboard_actuals
from rt_forecast_dashboard.ui import actuals as actuals_module


def test_fetch_dashboard_actuals_reads_compact_parquet(tmp_path, monkeypatch) -> None:
    path = tmp_path / "actuals_recent.parquet"
    pd.DataFrame(
        {
            "timestamp_utc": ["2026-10-01T10:00:00Z"],
            "collection_time_utc": ["2026-10-01T10:20:00Z"],
            "zone": ["BE"],
            "target": ["load"],
            "actual_mw": [101.0],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr(
        dashboard_actuals,
        "source_paths",
        lambda: {"entsoe_dashboard_actuals_url": path.as_uri()},
    )

    result = dashboard_actuals.fetch_dashboard_actuals()

    assert result["actual_mw"].tolist() == [101.0]
    assert result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-10-01T10:00:00Z"
    ]


def test_attach_display_actuals_uses_static_artifact(monkeypatch) -> None:
    forecasts = pd.DataFrame(
        {
            "zone": ["BE", "BE"],
            "target": ["load", "load"],
            "timestamp": pd.to_datetime(
                ["2026-10-01T10:00:00Z", "2026-10-01T11:00:00Z"], utc=True
            ),
            "actual_mw": [pd.NA, 99.0],
        }
    )
    artifact = pd.DataFrame(
        {
            "zone": ["BE", "BE"],
            "target": ["load", "load"],
            "timestamp": pd.to_datetime(
                ["2026-10-01T10:00:00Z", "2026-10-01T11:00:00Z"], utc=True
            ),
            "actual_mw": [101.0, 102.0],
            "collection_time_utc": pd.to_datetime(
                ["2026-10-01T12:00:00Z", "2026-10-01T12:00:00Z"], utc=True
            ),
        }
    )
    monkeypatch.setattr(actuals_module, "fetch_dashboard_actuals", lambda: artifact)

    result = actuals_module.attach_display_actuals(
        forecasts, now=pd.Timestamp("2026-10-01T12:30:00Z")
    )

    assert result["actual_mw"].tolist() == [101.0, 99.0]
    assert result.attrs["actual_artifact_age_minutes"] == 30.0
    assert result.attrs["actual_fetch_errors"] == []


def test_attach_display_actuals_keeps_stored_values_when_artifact_fails(monkeypatch) -> None:
    forecasts = pd.DataFrame(
        {
            "zone": ["BE"],
            "target": ["load"],
            "timestamp": pd.to_datetime(["2026-10-01T10:00:00Z"], utc=True),
            "actual_mw": [pd.NA],
        }
    )

    def fail():
        raise RuntimeError("artifact unavailable")

    monkeypatch.setattr(actuals_module, "fetch_dashboard_actuals", fail)
    result = actuals_module.attach_display_actuals(
        forecasts, now=pd.Timestamp("2026-10-01T12:30:00Z")
    )

    assert result["actual_mw"].isna().all()
    assert result.attrs["actual_fetch_errors"] == ["artifact unavailable"]
