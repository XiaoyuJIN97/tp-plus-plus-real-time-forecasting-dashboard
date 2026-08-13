from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.data.entsoe_client import EntsoeForecastClient


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

    zone_config = zones()
    entsoe = EntsoeForecastClient(demo_mode=False)
    keys = frame.loc[needs_actual, ["zone", "target"]].drop_duplicates()
    actual_frames = []
    errors = []
    for row in keys.itertuples(index=False):
        subset = frame.loc[needs_actual & frame["zone"].eq(row.zone) & frame["target"].eq(row.target)]
        if subset.empty or row.zone not in zone_config:
            continue
        start = subset["timestamp"].min()
        end = subset["timestamp"].max() + pd.Timedelta(hours=1)
        try:
            actual = entsoe.fetch_actuals(
                zone=row.zone,
                zone_code=zone_config[row.zone]["entsoe_code"],
                target=row.target,
                start=start,
                end=end,
            )
        except Exception as exc:
            message = str(exc).strip()
            if message:
                errors.append(f"{row.zone} {row.target}: {message}")
            continue
        actual["zone"] = row.zone
        actual["target"] = row.target
        actual_frames.append(actual)

    if not actual_frames:
        frame.attrs["actual_fetch_errors"] = errors
        return frame

    actuals = pd.concat(actual_frames, ignore_index=True).drop_duplicates(["zone", "target", "timestamp"])
    frame = frame.merge(actuals, on=["zone", "target", "timestamp"], how="left", suffixes=("", "_display"))
    frame["actual_mw"] = frame["actual_mw"].combine_first(frame["actual_mw_display"])
    frame.attrs["actual_fetch_errors"] = errors
    return frame.drop(columns=["actual_mw_display"])
