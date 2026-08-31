from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_ZONES = ("BE", "FR", "DE")
EXPECTED_TARGETS = ("load", "solar", "wind_onshore", "wind_offshore")
TARGET_MODELS = {
    "load": {"tso_reference", "persistence", "ridge_3mo_context", "chronos2_online"},
    "solar": {"tso_reference", "persistence", "chronos2_online", "xgboost_online"},
    "wind_onshore": {"tso_reference", "persistence", "chronos2_online", "xgboost_online"},
    "wind_offshore": {"tso_reference", "persistence", "chronos2_online", "xgboost_online"},
}
HORIZON_HOURS = 24


def _requested_models(value: str | None) -> set[str] | None:
    if not value:
        return None
    models = {item.strip() for item in value.split(",") if item.strip()}
    return models or None


def expected_groups(model_keys: set[str] | None = None) -> set[tuple[str, str, str]]:
    groups: set[tuple[str, str, str]] = set()
    for zone in EXPECTED_ZONES:
        for target in EXPECTED_TARGETS:
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
    except Exception as exc:
        return False, f"cannot read {path}: {exc}"

    for group, values in timestamps.items():
        counts[group] = len(values)

    incomplete = {group: count for group, count in counts.items() if count != HORIZON_HOURS}
    if incomplete:
        sample = ", ".join(f"{zone}/{target}/{model}={count}" for (zone, target, model), count in sorted(incomplete.items())[:8])
        return False, f"incomplete groups: {sample}"
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
