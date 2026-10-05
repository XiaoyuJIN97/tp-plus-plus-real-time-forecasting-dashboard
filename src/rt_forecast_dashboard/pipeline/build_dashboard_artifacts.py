from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from rt_forecast_dashboard.storage import ForecastStore


DEFAULT_RETENTION_DAYS = 90


def build_dashboard_forecasts(
    store: ForecastStore,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    actuals_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    forecasts = store.read_forecasts()
    if forecasts.empty:
        raise RuntimeError("No forecast rows are available for the dashboard artifact.")

    run_days = pd.to_datetime(forecasts["run_date"], errors="coerce")
    latest_day = run_days.max()
    if pd.isna(latest_day):
        raise RuntimeError("Forecast rows do not contain a valid run_date.")
    cutoff = latest_day.normalize() - pd.Timedelta(days=max(1, retention_days) - 1)
    recent = forecasts.loc[run_days.ge(cutoff)].copy()
    recent["run_date"] = recent["run_date"].astype(str)
    for column in ("timestamp", "run_at", "context_start", "context_end"):
        if column in recent.columns:
            recent[column] = pd.to_datetime(recent[column], utc=True, errors="coerce")

    if actuals_path is not None and actuals_path.exists():
        actuals = pd.read_parquet(actuals_path)
        timestamp_column = "timestamp_utc" if "timestamp_utc" in actuals.columns else "timestamp"
        actuals["timestamp"] = pd.to_datetime(actuals[timestamp_column], utc=True, errors="coerce")
        actuals = (
            actuals.dropna(subset=["zone", "target", "timestamp", "actual_mw"])
            .sort_values("collection_time_utc" if "collection_time_utc" in actuals.columns else "timestamp")
            .drop_duplicates(["zone", "target", "timestamp"], keep="last")
        )
        actual_lookup = actuals[["zone", "target", "timestamp", "actual_mw"]].rename(
            columns={"actual_mw": "artifact_actual_mw"}
        )
        recent = recent.merge(actual_lookup, on=["zone", "target", "timestamp"], how="left")
        stored = pd.to_numeric(recent.get("actual_mw"), errors="coerce")
        recent["actual_mw"] = stored.fillna(recent.pop("artifact_actual_mw"))

    sort_columns = [column for column in ["run_date", "zone", "target", "model", "timestamp"] if column in recent]
    recent = recent.sort_values(sort_columns).reset_index(drop=True)
    path = store.write_dashboard_forecasts(recent)
    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "retention_days": retention_days,
        "first_run_date": str(recent["run_date"].min()),
        "latest_run_date": str(recent["run_date"].max()),
        "rows": len(recent),
        "zones": int(recent["zone"].nunique()),
        "targets": int(recent["target"].nunique()),
        "models": int(recent["model"].nunique()),
    }
    metadata_path = path.with_name("forecasts_status.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact files used by the Streamlit dashboard.")
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument("--actuals", type=Path)
    args = parser.parse_args()
    path, metadata = build_dashboard_forecasts(
        ForecastStore(), retention_days=args.retention_days, actuals_path=args.actuals
    )
    print(f"Wrote {path} ({metadata['rows']:,} rows)")


if __name__ == "__main__":
    main()
