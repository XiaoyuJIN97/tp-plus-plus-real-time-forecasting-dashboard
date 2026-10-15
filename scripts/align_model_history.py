"""Audit and repair model histories using saved replays and archived inputs.

Mock rows keep their provenance. Valid existing model runs are never replaced.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from rt_forecast_dashboard.config import model_registry, zones
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.ui.analytics import valid_online_forecasts

# TabPFN is currently hidden in the online dashboard.
HIDDEN_HISTORY_MODELS = {"tabpfn_online"}
KEYS = ["run_date", "zone", "target", "model"]


def expected_groups(start: str, end: str) -> pd.DataFrame:
    rows = []
    registry = model_registry()
    for day in pd.date_range(start, end):
        for zone, config in zones().items():
            for target in config["targets"]:
                models = {
                    key for key, model in registry.items()
                    if model.get("enabled", False)
                    and not model.get("pipeline_generated", False)
                    and key not in HIDDEN_HISTORY_MODELS
                    and target in model.get("targets", [])
                }
                for model in models:
                    rows.append((day.date().isoformat(), zone, target, model))
    return pd.DataFrame(rows, columns=KEYS)


def missing_groups(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    expected = expected_groups(start, end)
    valid = valid_online_forecasts(frame)
    present = valid[KEYS].drop_duplicates() if not valid.empty else expected.iloc[:0]
    merged = expected.merge(present.assign(present=True), on=KEYS, how="left")
    return merged.loc[merged.present.isna(), KEYS]


def import_replay(store: ForecastStore, path: Path, start: str, end: str) -> int:
    replay = pd.read_parquet(path)
    replay["run_date"] = replay.run_date.astype(str)
    replay = replay[replay.run_date.between(start, end)].copy()
    replay["source"] = "historical_mock_replay"
    replay["history_origin"] = path.name
    replay["run_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    # Only import complete, correctly timed runs which the deployed store lacks.
    replay = valid_online_forecasts(replay)
    absent = missing_groups(store.read_forecasts(), start, end)
    replay = replay.merge(absent, on=KEYS, how="inner")
    for day, part in replay.groupby("run_date"):
        store.append_forecasts(part, str(day))
    return len(replay)


def repair_groups(store: ForecastStore, missing: pd.DataFrame, supplemental_archive: Path | None = None) -> None:
    import rt_forecast_dashboard.pipeline.run_daily as daily

    # Reuse parsed archives and model objects throughout a historical repair.
    # Loading the same yearly archive for every model/date is needlessly costly.
    entsoe = daily.EntsoeForecastClient()
    weather = daily.OpenMeteoClient()
    entsoe.archive._git_show = lru_cache(maxsize=256)(entsoe.archive._git_show)
    entsoe.archive._git_show_bytes = lru_cache(maxsize=128)(entsoe.archive._git_show_bytes)
    weather.archive._git_show = lru_cache(maxsize=32)(weather.archive._git_show)
    supplement = None
    if supplemental_archive is not None:
        from rt_forecast_dashboard.data.entsoe_realtime_archive import EntsoeRealtimeArchive
        supplement = EntsoeRealtimeArchive(root=supplemental_archive)
        # This option is a local saved archive, not a Git checkout.
        supplement._git_show = lambda path: None
        supplement._git_show_bytes = lambda path: None
    read_variable = entsoe.archive._read_variable
    earliest = pd.Timestamp(missing.run_date.min(), tz="UTC") - pd.Timedelta(days=97)
    latest = pd.Timestamp(missing.run_date.max(), tz="UTC") + pd.Timedelta(days=2)
    archived = {}

    def cached_variable(*, zone, variable, start, end):
        key = (zone, variable)
        if key not in archived:
            existing = read_variable(zone=zone, variable=variable, start=earliest, end=latest)
            if supplement is not None:
                older = supplement._read_variable(zone=zone, variable=variable, start=earliest, end=latest)
                if existing.empty:
                    existing = older
                elif not older.empty:
                    existing = pd.concat([older, existing], ignore_index=True).drop_duplicates("timestamp", keep="last")
            archived[key] = existing
        frame = archived[key]
        return frame[frame.timestamp.between(start, end, inclusive="left")].copy()

    entsoe.archive._read_variable = cached_variable
    original_entsoe, original_weather = daily.EntsoeForecastClient, daily.OpenMeteoClient
    daily.EntsoeForecastClient = lambda: entsoe
    daily.OpenMeteoClient = lambda: weather
    configuration = zones()
    original_zones = daily.zones
    try:
        for index, ((day, zone, target), group) in enumerate(missing.groupby(KEYS[:3]), 1):
            selected = dict(configuration[zone], targets=[target])
            daily.zones = lambda: {zone: selected}
            models = set(group.model)
            print(f"Repair {index}: {day} {zone} {target} {sorted(models)}", flush=True)
            result = daily.run_daily_forecast(pd.Timestamp(day).date(), model_keys=models, clear_issues=False)
            if not result.empty:
                result["source"] = "historical_mock_replay"
                result["history_origin"] = "archived_input_replay"
                store.append_forecasts(result, day)
            # Persist progress even if a later model or source fails.
    finally:
        daily.zones = original_zones
        daily.EntsoeForecastClient = original_entsoe
        daily.OpenMeteoClient = original_weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--import-replay", type=Path)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--supplemental-archive", type=Path, help="Saved ENTSO-E hourly archive for replay context.")
    args = parser.parse_args()
    store = ForecastStore()
    if args.import_replay:
        print(f"Imported {import_replay(store, args.import_replay, args.start, args.end):,} saved replay rows.", flush=True)
    missing = missing_groups(store.read_forecasts(), args.start, args.end)
    print(f"Missing or incorrectly timed model runs: {len(missing)}", flush=True)
    if args.repair and not missing.empty:
        repair_groups(store, missing, args.supplemental_archive)
        missing = missing_groups(store.read_forecasts(), args.start, args.end)
    report = store.data_dir / "diagnostics" / "model_history_missing.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(report, index=False)
    print(f"Remaining model runs: {len(missing)}. Report: {report}", flush=True)
    if not missing.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
