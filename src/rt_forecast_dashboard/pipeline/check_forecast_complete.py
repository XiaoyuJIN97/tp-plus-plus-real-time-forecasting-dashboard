from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from rt_forecast_dashboard.config import features, zones


TARGET_MODELS = {
    "load": {"tso_reference", "persistence", "ridge_3mo_context", "chronos2_online", "timesfm3_online"},
    "solar": {"tso_reference", "persistence", "chronos2_online", "timesfm3_online", "xgboost_online"},
    "wind_onshore": {"tso_reference", "persistence", "chronos2_online", "timesfm3_online", "xgboost_online"},
    "wind_offshore": {"tso_reference", "persistence", "chronos2_online", "timesfm3_online", "xgboost_online"},
}
HORIZON_HOURS = 24


def _requested_models(value: str | None) -> set[str] | None:
    if not value:
        return None
    models = {item.strip() for item in value.split(",") if item.strip()}
    return models or None


def expected_groups(model_keys: set[str] | None = None) -> set[tuple[str, str, str]]:
    groups: set[tuple[str, str, str]] = set()
    target_names = tuple(features())
    for zone, zone_config in zones().items():
        enabled_targets = set(zone_config.get("targets", target_names))
        for target in target_names:
            if target not in enabled_targets:
                continue
            for model in TARGET_MODELS[target]:
                if model_keys is None or model in model_keys:
                    groups.add((zone, target, model))
    return groups


def is_forecast_complete(path: Path, model_keys: set[str] | None = None) -> tuple[bool, str]:
    expected = expected_groups(model_keys)
    if not path.exists():
        return False, f"missing {path}"

    counts = {group: 0 for group in expected}
    timestamps: dict[tuple[str, str, str], set[str]] = {group: set() for group in expected}
    first_delivery: dict[tuple[str, str, str], datetime] = {}
    latest_context_end: dict[tuple[str, str, str], datetime] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"zone", "target", "model", "timestamp", "forecast_mw"}
            missing_columns = required.difference(reader.fieldnames or [])
            if missing_columns:
                return False, f"missing columns: {', '.join(sorted(missing_columns))}"
            for row in reader:
                group = (row.get("zone", ""), row.get("target", ""), row.get("model", ""))
                if group not in expected:
                    continue
                if not row.get("timestamp") or not row.get("forecast_mw"):
                    continue
                timestamps[group].add(str(row["timestamp"]))
                delivery = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                first_delivery[group] = min(first_delivery.get(group, delivery), delivery)
                if row.get("context_end"):
                    context_end = datetime.fromisoformat(str(row["context_end"]).replace("Z", "+00:00"))
                    latest_context_end[group] = max(latest_context_end.get(group, context_end), context_end)
    except Exception as exc:
        return False, f"cannot read {path}: {exc}"

    for group, values in timestamps.items():
        counts[group] = len(values)

    incomplete = {group: count for group, count in counts.items() if count != HORIZON_HOURS}
    if incomplete:
        sample = ", ".join(f"{zone}/{target}/{model}={count}" for (zone, target, model), count in sorted(incomplete.items())[:8])
        return False, f"incomplete groups: {sample}"
    stale = {
        group: (latest_context_end[group], delivery - timedelta(hours=1))
        for group, delivery in first_delivery.items()
        if group in latest_context_end and latest_context_end[group] < delivery - timedelta(hours=1)
    }
    if stale:
        sample = ", ".join(
            f"{zone}/{target}/{model} context_end={actual.isoformat()} expected>={expected_end.isoformat()}"
            for (zone, target, model), (actual, expected_end) in sorted(stale.items())[:8]
        )
        return False, f"stale context: {sample}"
    return True, f"complete {len(expected)} groups x {HORIZON_HOURS} hours"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a stored daily forecast file is complete.")
    parser.add_argument("--date", required=True, help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--data-dir", default="data", help="Dashboard data directory.")
    parser.add_argument("--models", default=None, help="Comma-separated model keys expected in this pass.")
    args = parser.parse_args()

    path = Path(args.data_dir) / "forecasts" / f"forecasts_{args.date}.csv"
    complete, reason = is_forecast_complete(path, _requested_models(args.models))
    print(reason)
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
