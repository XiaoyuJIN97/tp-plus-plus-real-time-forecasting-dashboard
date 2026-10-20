"""Replay pinned foundation models on saved TimesFM3 windows, without future actuals.

Run with the production checkout on PYTHONPATH. Inputs are collector archives;
forecasts are tagged historical_mock_replay, and failures remain explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.context_quality import repair_context
from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.data.entsoe_realtime_archive import ACTUAL_VARIABLES, FORECAST_VARIABLES
from rt_forecast_dashboard.data.weather_archive import TARGET_MAP
from rt_forecast_dashboard.models.foundation_online import FoundationOnlineAdapter, close_workers
from rt_forecast_dashboard.storage import ForecastStore


def hourly(path):
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    return frame.set_index("timestamp").value.resample("h").mean().dropna()


def load_series(inputs, history, zone, target, value):
    variable = (ACTUAL_VARIABLES if value == "actual_mw" else FORECAST_VARIABLES)[target]
    parts = []
    raw = inputs / "entsoe/data/raw" / zone / variable / "2026.csv"
    if raw.exists():
        parts.append(hourly(raw))
    # Retained realized outcomes cover observations pruned from the collector.
    old = history[(history.zone == zone) & (history.target == target)].copy()
    if value == "tso_forecast_mw":
        old = old[old.model.eq("timesfm3_online")]
    old[value] = pd.to_numeric(old[value], errors="coerce")
    old = old.dropna(subset=[value]).drop_duplicates("timestamp", keep="last")
    parts.append(old.set_index("timestamp")[value])
    current = inputs / "entsoe/data/hourly" / zone / variable / "2026.parquet"
    if current.exists():
        parts.append(hourly(current))
    result = pd.concat(parts)
    return result[~result.index.duplicated(keep="last")].sort_index().rename(value)


def load_weather(inputs, zone, target):
    path = inputs / "weather/data/raw" / zone / TARGET_MAP[target] / "2026.csv"
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    extra = inputs / "weather_supplement" / f"{zone}_{target}.csv"
    if extra.exists():
        supplement = pd.read_csv(extra)
        supplement["timestamp"] = pd.to_datetime(supplement.timestamp, utc=True)
        frame = pd.concat([frame, supplement], ignore_index=True)
    return frame.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()


def prepare(template, actual, tso, weather, covs):
    template = template.sort_values("timestamp").copy()
    if len(template) != 24 or template.timestamp.nunique() != 24:
        raise ValueError("Incomplete TimesFM3 reference horizon")
    cutoff = template.timestamp.min()
    expected = pd.date_range(cutoff, periods=24, freq="h")
    if not np.array_equal(expected.to_numpy(), template.timestamp.to_numpy()):
        raise ValueError("Noncontiguous reference horizon")
    index = pd.date_range(cutoff - pd.Timedelta(hours=2208), periods=2208, freq="h")
    context = pd.DataFrame({"actual_mw": actual.reindex(index), "tso_forecast_mw": tso.reindex(index)})
    context.index.name = "timestamp"
    context, quality = repair_context(context.reset_index(), expected_end=index[-1], context_hours=2208)
    # Interpolate past covariates separately; future values cannot fill the past.
    past_cov = weather.reindex(index).copy()
    past_cov["tso_forecast_mw"] = tso.reindex(index)
    future_cov = weather.reindex(expected).copy()
    future_cov["tso_forecast_mw"] = template.tso_forecast_mw.to_numpy(float)
    for frame in (past_cov, future_cov):
        if set(covs) - set(frame.columns):
            raise ValueError("Missing required weather columns")
        missing = frame[covs].isna()
        if missing.any().any():
            # Only isolated gaps; never extrapolate whole missing weather days.
            if missing.iloc[0].any() or missing.iloc[-1].any() or missing.sum().max() > 48:
                raise ValueError(f"Missing weather/TSO covariates: {int(missing.sum().sum())} cells")
        frame[covs] = frame[covs].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    x = context.actual_mw.to_numpy(np.float32)
    cov = pd.concat([past_cov[covs], future_cov[covs]]).to_numpy(np.float32).T
    if not np.isfinite(x).all() or not np.isfinite(cov).all():
        raise ValueError("Nonfinite prepared inputs")
    template["actual_mw"] = actual.reindex(expected).to_numpy()
    template["context_start"] = index[0]
    template["context_end"] = index[-1]
    template["context_hours"] = 2208
    template["context_missing_hours"] = quality.missing_hours
    template["context_imputed_hours"] = quality.imputed_hours
    template["context_imputation_method"] = quality.methods
    template["data_quality"] = quality.quality
    return x, cov, template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-07-24")
    parser.add_argument("--end", default="2026-10-19")
    parser.add_argument("--models", default="t0_gamma,timestral")
    parser.add_argument("--zones", default="BE,FR,DE")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in sorted((args.history / "forecasts").glob("forecasts_*.csv"))]
    compact = args.history / "dashboard/forecasts_recent.parquet"
    if compact.exists():
        frames.append(pd.read_parquet(compact))
    history = pd.concat(frames, ignore_index=True)
    history["timestamp"] = pd.to_datetime(history.timestamp, utc=True)
    history["run_date"] = history.run_date.astype(str)
    history = history.drop_duplicates(["run_date", "zone", "target", "model", "timestamp"], keep="last")
    reference = history[history.model.eq("timesfm3_online") & history.run_date.between(args.start, args.end)]
    audit, issues = [], []
    store = ForecastStore(args.output)
    for name in args.models.split(","):
        adapter = FoundationOnlineAdapter(name)
        for (zone, target), templates in reference.groupby(["zone", "target"]):
            if zone not in args.zones.split(",") or zone not in zones():
                continue
            actual = load_series(args.inputs, history, zone, target, "actual_mw")
            tso = load_series(args.inputs, history, zone, target, "tso_forecast_mw")
            weather = load_weather(args.inputs, zone, target)
            case, covs = covariates_for(target, zone, name + "_online")
            pending = []
            for run_date, template in templates.groupby("run_date"):
                saved = store.forecast_path(run_date)
                if saved.exists() and not args.preflight:
                    saved_rows = pd.read_csv(saved)
                    match = saved_rows[(saved_rows.zone == zone) & (saved_rows.target == target) & (saved_rows.model == name + "_online")]
                    if len(match) == 24 and match.timestamp.nunique() == 24 and np.isfinite(match.forecast_mw).all():
                        continue
                try:
                    x, cov, output = prepare(template, actual, tso, weather, covs)
                    pending.append((x, cov, output))
                    audit.append(dict(model=name, zone=zone, target=target, run_date=run_date, status="ready", imputed=int(output.context_imputed_hours.iloc[0])))
                except Exception as exc:
                    issues.append(dict(model=name, zone=zone, target=target, run_date=run_date, error=str(exc)))
            pd.DataFrame(issues).to_csv(args.output / "replay_issues.csv", index=False)
            pd.DataFrame(audit).to_csv(args.output / "input_audit.csv", index=False)
            if args.preflight:
                print(name, zone, target, "ready", len(pending), flush=True)
                continue
            started = time.monotonic()
            for offset in range(0, len(pending), args.batch_size):
                chunk = pending[offset:offset + args.batch_size]
                predictions = adapter.predict_batch([c[0] for c in chunk], [c[1] for c in chunk])
                for (_, _, output), pred in zip(chunk, predictions, strict=True):
                    output["model"] = name + "_online"
                    output["model_label"] = ("T0 Gamma" if name == "t0_gamma" else "Timestral") + " with covariates"
                    output["forecast_mw"] = pred
                    output["source"] = "historical_mock_replay"
                    output["run_at"] = datetime.now(UTC).isoformat()
                    output["covariate_case"] = case
                    store.append_forecasts(output, str(output.run_date.iloc[0]))
                print(name, zone, target, min(offset + len(chunk), len(pending)), "/", len(pending), round(time.monotonic()-started, 1), "s", flush=True)
        close_workers()
    (args.output / "replay_config.json").write_text(json.dumps({k: str(v) for k,v in vars(args).items()}, indent=2))
    print("Ready windows", len(audit), "blocked windows", len(issues), flush=True)


if __name__ == "__main__":
    main()
