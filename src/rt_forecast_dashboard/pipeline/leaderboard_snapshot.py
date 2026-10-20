from __future__ import annotations

import json

import pandas as pd

from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.ui.analytics import (
    online_rmae_leaderboard,
    online_win_rate_by_zone,
    valid_online_forecasts,
)


def build_daily_leaderboard(store: ForecastStore, forecasts: pd.DataFrame,
                            *, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Publish daily results, also accepting newly realized outcomes that day."""
    now = now or pd.Timestamp.now(tz="UTC")
    day = now.tz_convert("Europe/Brussels").date().isoformat()
    directory = store.dashboard_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "leaderboard_daily.parquet"
    status_path = directory / "leaderboard_daily_status.json"
    realized = forecasts.dropna(subset=["actual_mw"]) if "actual_mw" in forecasts else forecasts.iloc[:0]
    times = pd.to_datetime(realized.get("timestamp", pd.Series(dtype=str)), utc=True, errors="coerce")
    actuals_through = times.max().isoformat() if times.notna().any() else None
    actual_rows = len(realized)
    if path.exists() and status_path.exists():
        status = json.loads(status_path.read_text())
        if (status.get("snapshot_date") == day
                and status.get("actuals_through_utc") == actuals_through
                and status.get("actual_rows") == actual_rows):
            return pd.read_parquet(path)
    if not {"timestamp", "run_date", "model", "target", "zone", "forecast_mw"}.issubset(forecasts.columns):
        return pd.DataFrame()
    frame = valid_online_forecasts(forecasts)
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame.zone.isin(zones()) & frame.model.ne("tabpfn_online")].copy()
    detail, _ = online_rmae_leaderboard(frame)
    if detail.empty:
        return detail
    wins = online_win_rate_by_zone(frame)
    detail = detail.merge(wins[["zone", "target", "display_family", "win_rate"]],
                          on=["zone", "target", "display_family"], how="left")
    detail["snapshot_date"] = day
    temporary = path.with_suffix(".parquet.tmp")
    detail.to_parquet(temporary, index=False)
    temporary.replace(path)
    status = {"snapshot_date": day, "generated_at_utc": now.isoformat(),
              "first_run_date": str(frame.run_date.min()),
              "latest_run_date": str(frame.run_date.max()),
              "actuals_through_utc": actuals_through, "actual_rows": actual_rows}
    temporary_status = status_path.with_suffix(".json.tmp")
    temporary_status.write_text(json.dumps(status, indent=2) + "\n")
    temporary_status.replace(status_path)
    return detail
