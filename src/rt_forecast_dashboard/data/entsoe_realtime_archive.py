from __future__ import annotations

from pathlib import Path
from io import StringIO
import os
import subprocess

import pandas as pd

from rt_forecast_dashboard.config import source_paths


ACTUAL_VARIABLES = {
    "load": "actual_load",
    "solar": "actual_solar_generation",
    "wind_onshore": "actual_onshore_wind_generation",
    "wind_offshore": "actual_offshore_wind_generation",
}

FORECAST_VARIABLES = {
    "load": "forecast_load",
    "solar": "forecast_solar_generation",
    "wind_onshore": "forecast_onshore_wind_generation",
    "wind_offshore": "forecast_offshore_wind_generation",
}


class EntsoeRealtimeArchive:
    def __init__(self, root: str | Path | None = None) -> None:
        paths = source_paths()
        configured = paths.get("entsoe_realtime_data")
        self.root = Path(root or configured) if root or configured else None
        self.git_ref = paths.get("entsoe_realtime_git_ref")
        self._available_cache: bool | None = None
        self._manifest_cache: pd.DataFrame | None = None

    @property
    def available(self) -> bool:
        if self._available_cache is not None:
            return self._available_cache
        if not self.root:
            self._available_cache = False
            return False
        self._available_cache = bool(self._git_show("data/update_manifest.csv") or (self.root / "data" / "update_manifest.csv").exists())
        return self._available_cache

    def fetch_actuals(self, *, zone: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        variable = ACTUAL_VARIABLES[target]
        return self._read_variable(zone=zone, variable=variable, start=start, end=end).rename(columns={"value": "actual_mw"})

    def fetch_forecast(self, *, zone: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        variable = FORECAST_VARIABLES[target]
        return self._read_variable(zone=zone, variable=variable, start=start, end=end).rename(columns={"value": "tso_forecast_mw"})

    def _read_variable(self, *, zone: str, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.available or self.root is None:
            return pd.DataFrame(columns=["timestamp", "value"])

        start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        raw = self._read_raw_variable(zone=zone, variable=variable, start=start, end=end)
        manifest = self._read_manifest()
        manifest = manifest[(manifest["country"].eq(zone)) & (manifest["variable"].eq(variable))].copy()
        if manifest.empty:
            return raw

        manifest["collection_time_utc"] = pd.to_datetime(manifest["collection_time_utc"], utc=True)
        manifest["window_start_utc"] = pd.to_datetime(manifest["window_start_utc"], utc=True)
        manifest["window_end_utc"] = pd.to_datetime(manifest["window_end_utc"], utc=True)
        overlapping = manifest[(manifest["window_start_utc"].le(end)) & (manifest["window_end_utc"].ge(start))].copy()
        if overlapping.empty:
            return raw

        parts = []
        for row in overlapping.sort_values("collection_time_utc", ascending=False).head(5).itertuples(index=False):
            frame = self._read_snapshot_csv(row.path)
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
            frame["collection_time_utc"] = pd.to_datetime(frame["collection_time_utc"], utc=True)
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
            parts.append(frame[["timestamp", "value", "collection_time_utc"]])
            if not frame.empty:
                break

        if not parts:
            return raw

        data = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
        data = data.sort_values(["timestamp", "collection_time_utc"]).drop_duplicates("timestamp", keep="last")
        updates = self._to_hourly(data[["timestamp", "value"]])
        if raw.empty:
            return updates
        combined = pd.concat([raw, updates], ignore_index=True)
        return combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    def _read_manifest(self) -> pd.DataFrame:
        if self._manifest_cache is not None:
            return self._manifest_cache.copy()
        if not self.root:
            self._manifest_cache = pd.DataFrame()
            return self._manifest_cache.copy()
        manifest_path = self.root / "data" / "update_manifest.csv"
        manifest_text = self._git_show("data/update_manifest.csv")
        if manifest_text:
            self._manifest_cache = pd.read_csv(StringIO(manifest_text))
        elif manifest_path.exists():
            self._manifest_cache = pd.read_csv(manifest_path)
        else:
            self._manifest_cache = pd.DataFrame()
        return self._manifest_cache.copy()

    def _to_hourly(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        hourly = (
            frame.sort_values("timestamp")
            .set_index("timestamp")["value"]
            .resample("h")
            .mean()
            .dropna()
            .reset_index()
        )
        return hourly[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)

    def _read_snapshot_csv(self, path: str) -> pd.DataFrame:
        snapshot_text = self._git_show(path)
        if snapshot_text:
            return pd.read_csv(StringIO(snapshot_text), usecols=["collection_time_utc", "timestamp_utc", "value"])
        if not self.root:
            return pd.DataFrame()
        snapshot_path = self.root / path
        if not snapshot_path.exists():
            return pd.DataFrame()
        return pd.read_csv(snapshot_path, usecols=["collection_time_utc", "timestamp_utc", "value"])

    def _read_raw_variable(self, *, zone: str, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frames = []
        for year in range(start.year, end.year + 1):
            path = f"data/raw/{zone}/{variable}/{year}.csv"
            text = self._git_show(path)
            if text:
                frame = pd.read_csv(StringIO(text))
            elif self.root and (self.root / path).exists():
                frame = pd.read_csv(self.root / path)
            else:
                continue
            timestamp_col = "timestamp_utc" if "timestamp_utc" in frame.columns else frame.columns[0]
            value_col = "value" if "value" in frame.columns else frame.select_dtypes(include="number").columns[-1]
            frame = frame[[timestamp_col, value_col]].rename(columns={timestamp_col: "timestamp", value_col: "value"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "value"])
        return self._to_hourly(pd.concat(frames, ignore_index=True))

    def _git_show(self, path: str) -> str | None:
        if not self.root or not self.git_ref:
            return None
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), "show", f"{self.git_ref}:{path}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except Exception:
            return None
        return result.stdout
