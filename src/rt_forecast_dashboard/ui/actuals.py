from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.data.dashboard_actuals import fetch_dashboard_actuals


def attach_display_actuals(forecasts: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Attach ENTSO-E realized values for visualization without using them as forecast inputs."""
    if forecasts.empty:
        frame = forecasts.copy()
        frame.attrs["actual_fetch_errors"] = []
        return frame

    frame = forecasts.copy()
    frame.attrs["actual_fetch_errors"] = []
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    now = now or pd.Timestamp.now(tz="UTC")
    actual_cutoff = now - pd.Timedelta(minutes=30)
    needs_actual = frame["actual_mw"].isna() & frame["timestamp"].lt(actual_cutoff)
    if not needs_actual.any():
        return frame

    try:
        actuals = fetch_dashboard_actuals()
    except Exception as exc:
        frame.attrs["actual_fetch_errors"] = [str(exc).strip()]
        return frame

    artifact_generated_at = actuals["collection_time_utc"].max()
    artifact_age_minutes = None
    if pd.notna(artifact_generated_at):
        artifact_age_minutes = max(
            0.0, (now - artifact_generated_at).total_seconds() / 60
        )
    needed_keys = frame.loc[needs_actual, ["zone", "target"]].drop_duplicates()
    actuals = actuals.merge(needed_keys, on=["zone", "target"], how="inner")
    actuals = actuals.drop_duplicates(["zone", "target", "timestamp"], keep="last")
    frame = frame.merge(actuals, on=["zone", "target", "timestamp"], how="left", suffixes=("", "_display"))
    frame["actual_mw"] = frame["actual_mw"].combine_first(frame["actual_mw_display"])
    frame.attrs["actual_fetch_errors"] = []
    frame.attrs["actual_artifact_generated_at"] = artifact_generated_at
    if artifact_age_minutes is not None:
        frame.attrs["actual_artifact_age_minutes"] = artifact_age_minutes
    return frame.drop(columns=["actual_mw_display", "collection_time_utc"])
