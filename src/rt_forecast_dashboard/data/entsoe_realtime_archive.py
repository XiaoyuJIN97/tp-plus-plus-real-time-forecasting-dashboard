from __future__ import annotations

from pathlib import Path
from io import BytesIO, StringIO
import os
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
        self.raw_base = (paths.get("entsoe_realtime_raw_base") or "").rstrip("/")
        self.hourly_dir = Path(paths.get("entsoe_realtime_hourly_dir") or "data/hourly")
        self.prefer_local = os.getenv("ENTSOE_REALTIME_PREFER_LOCAL", "false").lower() in {"1", "true", "yes"}
        self._available_cache: bool | None = None
        self._manifest_cache: pd.DataFrame | None = None

    @property
    def available(self) -> bool:
        if self._available_cache is not None:
            return self._available_cache
        if not self.root and not self.raw_base:
            self._available_cache = False
            return False
        local_manifest_exists = bool(self.root and (self.root / "data" / "update_manifest.csv").exists())
        local_hourly_exists = bool(self.root and self._local_hourly_base().exists())
        self._available_cache = bool(self._git_show("data/update_manifest.csv") or local_manifest_exists or local_hourly_exists)
        return self._available_cache

    def fetch_actuals(self, *, zone: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        variable = ACTUAL_VARIABLES[target]
        return self._read_variable(zone=zone, variable=variable, start=start, end=end).rename(columns={"value": "actual_mw"})

    def fetch_forecast(self, *, zone: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        variable = FORECAST_VARIABLES[target]
        return self._read_variable(zone=zone, variable=variable, start=start, end=end).rename(columns={"value": "tso_forecast_mw"})

    def _read_variable(self, *, zone: str, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame(columns=["timestamp", "value"])

        start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        hourly = self._read_hourly_variable(zone=zone, variable=variable, start=start, end=end)
        legacy = self._read_legacy_variable(zone=zone, variable=variable, start=start, end=end)
        if hourly.empty:
            return legacy
        if legacy.empty:
            return hourly
        return self._merge_sources(legacy, hourly)

    def _read_legacy_variable(self, *, zone: str, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raw = self._read_raw_variable(zone=zone, variable=variable, start=start, end=end)
        manifest = self._read_manifest()
        if not {"country", "variable"}.issubset(manifest.columns):
            return raw
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
        return self._merge_sources(raw, updates)

    def _merge_sources(self, older: pd.DataFrame, newer: pd.DataFrame) -> pd.DataFrame:
        older = older.copy()
        newer = newer.copy()
        older["_source_rank"] = 0
        newer["_source_rank"] = 1
        combined = pd.concat([older, newer], ignore_index=True)
        combined = combined.sort_values(["timestamp", "_source_rank"]).drop_duplicates("timestamp", keep="last")
        return combined.drop(columns="_source_rank").sort_values("timestamp").reset_index(drop=True)

    def _read_hourly_variable(self, *, zone: str, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frames = []
        for year in range(start.year, end.year + 1):
            path = self._hourly_path(zone, variable, year)
            frame = self._read_hourly_parquet(path)
            if frame.empty:
                continue
            if "timestamp_utc" not in frame.columns or "value" not in frame.columns:
                continue
            columns = ["timestamp_utc", "value"]
            if "collection_time_utc" in frame.columns:
                columns.append("collection_time_utc")
            frame = frame[columns].rename(columns={"timestamp_utc": "timestamp"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
            if "collection_time_utc" in frame.columns:
                frame["collection_time_utc"] = pd.to_datetime(frame["collection_time_utc"], utc=True, errors="coerce")
                frame = frame.sort_values(["timestamp", "collection_time_utc"]).drop_duplicates("timestamp", keep="last")
            frames.append(frame[["timestamp", "value"]])
        if not frames:
            return pd.DataFrame(columns=["timestamp", "value"])
        return self._to_hourly(pd.concat(frames, ignore_index=True))

    def _read_manifest(self) -> pd.DataFrame:
        if self._manifest_cache is not None:
            return self._manifest_cache.copy()
        manifest_path = self.root / "data" / "update_manifest.csv" if self.root else None
        manifest_text = self._git_show("data/update_manifest.csv")
        if manifest_text:
            self._manifest_cache = pd.read_csv(StringIO(manifest_text))
        elif manifest_path and manifest_path.exists():
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

    def _read_hourly_parquet(self, path: str) -> pd.DataFrame:
        raw_bytes = None if Path(path).is_absolute() else self._git_show_bytes(path)
        if raw_bytes is not None:
            return pd.read_parquet(BytesIO(raw_bytes))
        if not self.root and not Path(path).is_absolute():
            return pd.DataFrame()
        snapshot_path = Path(path) if Path(path).is_absolute() else self.root / path
        if not snapshot_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(snapshot_path)

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
        if not self.prefer_local:
            raw_text = self._read_raw_url(path)
            if raw_text is not None:
                return raw_text
        local_text = self._read_local_git(path)
        if local_text is not None:
            return local_text
        return self._read_raw_url(path)

    def _git_show_bytes(self, path: str) -> bytes | None:
        if not self.prefer_local:
            raw_bytes = self._read_raw_url_bytes(path)
            if raw_bytes is not None:
                return raw_bytes
        local_bytes = self._read_local_git_bytes(path)
        if local_bytes is not None:
            return local_bytes
        return self._read_raw_url_bytes(path)

    def _read_local_git(self, path: str) -> str | None:
        raw = self._read_local_git_bytes(path)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _read_local_git_bytes(self, path: str) -> bytes | None:
        if not self.root or not self.git_ref:
            return None
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), "show", f"{self.git_ref}:{path}"],
                check=True,
                capture_output=True,
                timeout=3,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            return result.stdout
        except Exception:
            return None

    def _read_raw_url(self, path: str) -> str | None:
        if not self.raw_base:
            return None
        url = f"{self.raw_base}/{path}"
        try:
            request = Request(url, headers={"User-Agent": "tp-plus-plus-real-time-forecasting-dashboard"})
            with urlopen(request, timeout=10) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError):
            return None

    def _read_raw_url_bytes(self, path: str) -> bytes | None:
        if not self.raw_base:
            return None
        url = f"{self.raw_base}/{path}"
        try:
            request = Request(url, headers={"User-Agent": "tp-plus-plus-real-time-forecasting-dashboard"})
            with urlopen(request, timeout=10) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError):
            return None

    def _local_hourly_base(self) -> Path:
        if self.hourly_dir.is_absolute():
            return self.hourly_dir
        if self.root:
            return self.root / self.hourly_dir
        return self.hourly_dir

    def _hourly_path(self, zone: str, variable: str, year: int) -> str:
        path = self.hourly_dir / zone / variable / f"{year}.parquet"
        return str(path)
