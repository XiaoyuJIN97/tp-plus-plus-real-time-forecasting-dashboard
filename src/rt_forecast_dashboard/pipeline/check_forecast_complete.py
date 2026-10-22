from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rt_forecast_dashboard.config import features, zones, model_registry, model_supports_scope
from rt_forecast_dashboard.time_utils import forecast_horizon_hours




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
            for model, config in model_registry().items():
                if (config.get('enabled', False) and not config.get('pipeline_generated', False)
                        and (model_keys is not None or config.get('required_for_completeness', True))
                        and model_supports_scope(config, zone, target)
                        and (model_keys is None or model in model_keys)):
                    groups.add((zone, target, model))
    return groups


def is_forecast_complete(path: Path, model_keys: set[str] | None = None) -> tuple[bool, str]:
    expected = expected_groups(model_keys)
    if not path.exists():
        return False, f"missing {path}"

    counts = {group: 0 for group in expected}
    timestamps = {group: set() for group in expected}
    run_dates = {}
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
                delivery = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                if delivery.tzinfo is None:
                    raise ValueError('Delivery timestamps must include a timezone')
                delivery = delivery.astimezone(UTC)
                timestamps[group].add(delivery)
                day = row.get('run_date') or path.stem[-10:]
                if group in run_dates and run_dates[group] != day:
                    raise ValueError('Mixed run dates within a model group')
                run_dates[group] = day
                first_delivery[group] = min(first_delivery.get(group, delivery), delivery)
                if row.get("context_end"):
                    context_end = datetime.fromisoformat(str(row["context_end"]).replace("Z", "+00:00"))
                    latest_context_end[group] = max(latest_context_end.get(group, context_end), context_end)
    except Exception as exc:
        return False, f"cannot read {path}: {exc}"

    for group, values in timestamps.items():
        counts[group] = len(values)

    incomplete = {group: count for group, count in counts.items()
                  if count != forecast_horizon_hours(run_dates.get(group, path.stem[-10:]))}
    if incomplete:
        listed = ", ".join(f"{zone}/{target}/{model}={count}" for (zone, target, model), count in sorted(incomplete.items()))
        return False, f"incomplete groups ({len(incomplete)}): {listed}"
    if any(any(b-a != timedelta(hours=1) for a,b in zip(sorted(values),sorted(values)[1:]))
           for values in timestamps.values()):
        return False, 'noncontiguous hourly delivery grid'
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
    return True, f"complete {len(expected)} groups; cutoff-to-cutoff hourly windows"


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
