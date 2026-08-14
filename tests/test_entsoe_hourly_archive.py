from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.data.entsoe_realtime_archive import EntsoeRealtimeArchive


def test_hourly_parquet_archive_overrides_legacy_raw_rows(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "BE" / "actual_load"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp_utc": ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z"],
            "value": [100.0, 101.0],
        }
    ).to_csv(raw_dir / "2026.csv", index=False)

    hourly_dir = tmp_path / "data" / "hourly" / "BE" / "actual_load"
    hourly_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-08-01T01:00:00Z",
                "2026-08-01T01:00:00Z",
                "2026-08-01T02:00:00Z",
            ],
            "country": ["BE", "BE", "BE"],
            "variable": ["actual_load", "actual_load", "actual_load"],
            "value": [201.0, 202.0, 203.0],
            "collection_time_utc": [
                "2026-08-01T02:00:00Z",
                "2026-08-01T03:00:00Z",
                "2026-08-01T03:00:00Z",
            ],
        }
    ).to_parquet(hourly_dir / "2026.parquet", index=False)

    archive = EntsoeRealtimeArchive(root=tmp_path)
    frame = archive.fetch_actuals(
        zone="BE",
        target="load",
        start=pd.Timestamp("2026-08-01T00:00:00Z"),
        end=pd.Timestamp("2026-08-01T03:00:00Z"),
    )

    assert frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-08-01T00:00:00Z",
        "2026-08-01T01:00:00Z",
        "2026-08-01T02:00:00Z",
    ]
    assert frame["actual_mw"].tolist() == [100.0, 202.0, 203.0]
