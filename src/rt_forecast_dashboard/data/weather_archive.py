from __future__ import annotations

from io import StringIO
from pathlib import Path
import os
import subprocess

import pandas as pd

from rt_forecast_dashboard.config import source_paths


TARGET_MAP = {
    "load": "load",
    "solar": "solar",
    "wind_onshore": "onshore",
    "wind_offshore": "offshore",
}


class OpenMeteoRealtimeArchive:
    def __init__(self, root: str | Path | None = None) -> None:
        paths = source_paths()
        configured = paths.get("open_meteo_realtime_data")
        self.root = Path(root or configured) if root or configured else None
        self.git_ref = paths.get("open_meteo_realtime_git_ref")

    @property
    def available(self) -> bool:
        if not self.root:
            return False
        return (self.root / "data" / "raw").exists() or bool(self._git_show("data/update_manifest.csv"))

    def fetch_weather(self, *, zone: str, target: str, start: pd.Timestamp, hours: int) -> pd.DataFrame:
        if not self.available or self.root is None:
            return pd.DataFrame()
        start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        end = start + pd.Timedelta(hours=hours)
        archive_target = TARGET_MAP[target]
        frames = []
        for year in range(start.year, end.year + 1):
            path = f"data/raw/{zone}/{archive_target}/{year}.csv"
            text = self._git_show(path)
            if text:
                frame = pd.read_csv(StringIO(text))
            elif (self.root / path).exists():
                frame = pd.read_csv(self.root / path)
            else:
                continue
            if "timestamp_utc" not in frame.columns:
                continue
            frame = frame.rename(columns={"timestamp_utc": "timestamp"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        drop_columns = ["country", "target", "source", "updated_at_utc"]
        data = data.drop(columns=[column for column in drop_columns if column in data.columns])
        return data.reset_index(drop=True)

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
