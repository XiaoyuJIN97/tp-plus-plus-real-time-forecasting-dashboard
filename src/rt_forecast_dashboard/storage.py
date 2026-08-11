from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rt_forecast_dashboard.config import load_settings


class ForecastStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or load_settings().data_dir
        for folder in ("forecasts", "raw", "issues", "backfill"):
            (self.data_dir / folder).mkdir(parents=True, exist_ok=True)

    def forecast_path(self, run_date: str) -> Path:
        return self.data_dir / "forecasts" / f"forecasts_{run_date}.csv"

    def raw_path(self, run_date: str, zone: str, target: str) -> Path:
        return self.data_dir / "raw" / f"raw_{run_date}_{zone}_{target}.csv"

    def issue_path(self) -> Path:
        return self.data_dir / "issues" / "issue_log.csv"

    def append_forecasts(self, frame: pd.DataFrame, run_date: str) -> None:
        path = self.forecast_path(run_date)
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp", "run_at"])
            replace_keys = ["run_date", "zone", "target", "model"]
            if all(column in existing.columns for column in replace_keys) and all(column in frame.columns for column in replace_keys):
                incoming_keys = frame[replace_keys].drop_duplicates()
                existing = existing.merge(incoming_keys.assign(_replace=True), on=replace_keys, how="left")
                existing = existing[existing["_replace"].isna()].drop(columns="_replace")
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(["run_date", "zone", "target", "model", "timestamp"], keep="last")
        frame.to_csv(path, index=False)

    def read_forecasts(self) -> pd.DataFrame:
        files = sorted((self.data_dir / "forecasts").glob("forecasts_*.csv"))
        if not files:
            return pd.DataFrame()
        frame = pd.concat((pd.read_csv(path, parse_dates=["timestamp", "run_at"]) for path in files), ignore_index=True)
        defaults = {
            "source": "legacy",
            "covariate_case": "legacy",
            "context_hours": 0,
            "context_start": pd.NaT,
            "context_end": pd.NaT,
            "actual_mw": pd.NA,
        }
        for column, value in defaults.items():
            if column not in frame.columns:
                frame[column] = value
        frame["source"] = frame["source"].fillna("legacy")
        frame["context_hours"] = pd.to_numeric(frame["context_hours"], errors="coerce").fillna(0).astype(int)
        return frame

    def write_raw(self, frame: pd.DataFrame, run_date: str, zone: str, target: str) -> None:
        frame.to_csv(self.raw_path(run_date, zone, target), index=False)

    def log_issue(self, *, run_date: str, zone: str, target: str, stage: str, message: str, context: dict[str, Any] | None = None) -> None:
        row = pd.DataFrame(
            [
                {
                    "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "run_date": run_date,
                    "zone": zone,
                    "target": target,
                    "stage": stage,
                    "message": message,
                    "context": context or {},
                    "status": "open",
                }
            ]
        )
        path = self.issue_path()
        if path.exists():
            row = pd.concat([pd.read_csv(path), row], ignore_index=True)
        row.to_csv(path, index=False)

    def read_issues(self) -> pd.DataFrame:
        path = self.issue_path()
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=["logged_at"])

    def clear_issues_for_run(self, run_date: str) -> None:
        path = self.issue_path()
        if not path.exists():
            return
        issues = pd.read_csv(path)
        issues = issues[issues["run_date"].astype(str) != run_date]
        issues.to_csv(path, index=False)
