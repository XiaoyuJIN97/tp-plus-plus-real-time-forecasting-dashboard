from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rt_forecast_dashboard.config import load_yaml, model_registry, model_supports_scope
from rt_forecast_dashboard.storage import ForecastStore


ENSEMBLE_MODEL_KEY = "ensemble_selected"
OFFLINE_WEIGHTS_PATH = Path("docs/new_zone_ensemble_weights.csv")


def _model_columns(registry=None) -> dict[str, str]:
    """Registry-driven membership; baselines/experimental models opt out explicitly."""
    return {
        key: config.get("ensemble_label", config.get("label", key)).removesuffix(" with covariates")
        for key, config in (model_registry() if registry is None else registry).items()
        if config.get("enabled", False)
        and not config.get("pipeline_generated", False)
        and config.get("ensemble_include", True)
        and key != ENSEMBLE_MODEL_KEY
    }


def _component_columns(target: str, include_tso: bool, zone: str | None = None, registry=None) -> list[str]:
    registry = model_registry() if registry is None else registry
    return [key for key in _model_columns(registry)
            if target in registry[key].get("targets", [])
            and (zone is None or model_supports_scope(registry[key], zone, target))
            and (key != "tso_reference" or include_tso)]


def _fit_weights(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    *,
    regularization: float,
    maximum_weight: float,
) -> np.ndarray:
    model_count = forecasts.shape[1]
    equal = np.full(model_count, 1.0 / model_count)
    scale = float(np.std(actuals))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.mean(np.abs(actuals))), 1.0)
    x = forecasts / scale
    y = actuals / scale

    def objective(weights: np.ndarray) -> float:
        residual = y - x @ weights
        return float(np.mean(residual**2) + regularization * np.sum((weights - equal) ** 2))

    result = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(0.0, maximum_weight)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        print(f"Ensemble weight optimization fell back to equal weights: {result.message}")
        return equal
    weights = np.clip(result.x, 0.0, maximum_weight)
    return weights / weights.sum()


def _load_actuals(path: Path) -> pd.DataFrame:
    actuals = pd.read_parquet(path)
    timestamp_column = "timestamp_utc" if "timestamp_utc" in actuals.columns else "timestamp"
    actuals["timestamp"] = pd.to_datetime(actuals[timestamp_column], utc=True, errors="coerce")
    if "collection_time_utc" in actuals.columns:
        actuals["collection_time_utc"] = pd.to_datetime(actuals["collection_time_utc"], utc=True, errors="coerce")
        actuals = actuals.sort_values("collection_time_utc")
    return (
        actuals.dropna(subset=["zone", "target", "timestamp", "actual_mw"])
        .drop_duplicates(["zone", "target", "timestamp"], keep="last")
        [["zone", "target", "timestamp", "actual_mw"]]
    )


def _forecast_panel(forecasts: pd.DataFrame, actuals: pd.DataFrame, model_columns=None) -> pd.DataFrame:
    components = set(_model_columns() if model_columns is None else model_columns)
    frame = forecasts[forecasts["model"].isin(components)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["run_at"] = pd.to_datetime(frame["run_at"], utc=True, errors="coerce")
    frame = frame.sort_values("run_at").drop_duplicates(
        ["run_date", "zone", "target", "timestamp", "model"], keep="last"
    )
    panel = frame.pivot(
        index=["run_date", "zone", "target", "timestamp"],
        columns="model",
        values="forecast_mw",
    ).reset_index()
    panel = panel.merge(actuals, on=["zone", "target", "timestamp"], how="left")
    return panel.sort_values(["zone", "target", "run_date", "timestamp"]).reset_index(drop=True)


def _complete_run(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty or not set(columns).issubset(frame.columns):
        return pd.DataFrame()
    complete = frame.dropna(subset=columns).sort_values("timestamp").copy()
    if not np.isfinite(complete[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        return pd.DataFrame()
    if (len(complete) != 24 or complete["timestamp"].nunique() != 24
            or not complete["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()):
        return pd.DataFrame()
    complete["horizon"] = np.arange(1, 25)
    return complete


def _band_for_horizon(horizon: int, bands: list[list[int]]) -> int:
    for index, (start, end) in enumerate(bands):
        if start <= horizon <= end:
            return index
    raise ValueError(f"Horizon {horizon} is not covered by the configured bands.")


def _offline_fallback_weights(
    weights: pd.DataFrame,
    *,
    zone: str,
    target: str,
    band: list[int],
    components: list[str],
    include_tso: bool,
    model_columns=None,
) -> np.ndarray | None:
    if weights.empty:
        return None
    band_label = f"{band[0]:02d}-{band[1]:02d}"
    component_set = "With TSO" if include_tso else "Without TSO"
    selected = weights[
        weights["zone"].eq(zone)
        & weights["target"].eq(target)
        & weights["horizon_band"].astype(str).eq(band_label)
        & weights["component_set"].eq(component_set)
    ]
    by_model = selected.set_index("model")["mean_weight"] if not selected.empty else pd.Series(dtype=float)
    model_columns = _model_columns() if model_columns is None else model_columns
    labels = [model_columns[component] for component in components]
    if not set(labels).issubset(by_model.index):
        return None
    result = by_model.reindex(labels).to_numpy(float)
    if not np.all(np.isfinite(result)) or result.sum() <= 0:
        return None
    return result / result.sum()


def build_selected_ensembles(
    store: ForecastStore,
    *,
    actuals_path: Path,
    run_dates: list[str],
    source: str = "online",
    offline_weights_path: Path = OFFLINE_WEIGHTS_PATH,
    only_missing: bool = False,
    refresh_dates: set[str] | None = None,
    model_registry_override: dict | None = None,
) -> pd.DataFrame:
    config = load_yaml("ensemble_registry.yml")
    settings = config["settings"]
    selections = config["selections"]
    forecasts = store.read_forecasts()
    for column in ("timestamp", "run_at"):
        forecasts[column] = pd.to_datetime(forecasts[column], utc=True, format="mixed", errors="coerce")
    forecasts["run_date"] = forecasts["run_date"].astype(str)
    actuals = _load_actuals(actuals_path)
    # Preserve outcomes older than the collector's shorter retained snapshot.
    if "actual_mw" in forecasts:
        stored_actuals = forecasts.dropna(subset=["actual_mw"]).sort_values("run_at").drop_duplicates(
            ["zone", "target", "timestamp"], keep="last"
        )[["zone", "target", "timestamp", "actual_mw"]]
        actuals = pd.concat([stored_actuals, actuals], ignore_index=True).drop_duplicates(
            ["zone", "target", "timestamp"], keep="last"
        )
    offline_weights = pd.read_csv(offline_weights_path) if offline_weights_path.exists() else pd.DataFrame()
    registry = model_registry() if model_registry_override is None else model_registry_override
    model_columns = _model_columns(registry)
    panel = _forecast_panel(forecasts, actuals, model_columns)
    prepared_cache: dict[tuple, list[pd.DataFrame]] = {}
    forecast_runs = {
        key: frame for key, frame in forecasts.groupby(["run_date", "zone", "target"])
    }
    panel_series = {key: frame for key, frame in panel.groupby(["zone", "target"])}
    outputs: list[pd.DataFrame] = []
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    for run_date in sorted(set(run_dates)):
        for zone, target_configs in selections.items():
            for target, selection in target_configs.items():
                method = selection["method"]
                include_tso = bool(selection["include_tso"])
                requested = _component_columns(target, include_tso, zone, registry)
                series = panel_series.get((zone, target), panel.iloc[:0])
                run_panel = series[series["run_date"].eq(run_date)]
                if run_panel.empty:
                    continue
                # Historical additions and temporary outages must not block all forecasts.
                # Include every eligible model with a complete, finite 24-hour run.
                components = [key for key in requested if not _complete_run(run_panel, [key]).empty]
                missing = sorted(set(requested) - set(components))
                if missing:
                    print(f"Ensemble {run_date} {zone} {target}: unavailable components {missing}")
                if len(components) < 2:
                    print(f"Skip ensemble {run_date} {zone} {target}: fewer than two complete components.")
                    continue
                component_signature = ",".join(components)
                current = _complete_run(run_panel, components)
                if current.empty:
                    print(f"Skip ensemble {run_date} {zone} {target}: component forecasts are incomplete.")
                    continue

                if only_missing and run_date not in (refresh_dates or set()):
                    run_forecasts = forecast_runs.get((run_date, zone, target), forecasts.iloc[:0])
                    saved = run_forecasts[run_forecasts["model"].eq(ENSEMBLE_MODEL_KEY)].copy()
                    saved["timestamp"] = pd.to_datetime(saved["timestamp"], utc=True)
                    saved = _complete_run(saved, ["forecast_mw"])
                    context_end = pd.to_datetime(saved.get("context_end"), utc=True)
                    if (not saved.empty
                            and saved["timestamp"].tolist() == current["timestamp"].tolist()
                            and context_end.eq(current["timestamp"].min() - pd.Timedelta(hours=1)).all()
                            and np.isfinite(saved["forecast_mw"].to_numpy(float)).all()
                            and "ensemble_components" in saved
                            and saved["ensemble_components"].eq(component_signature).all()):
                        continue

                values = current[components].to_numpy(float)
                weights_by_band: dict[int, np.ndarray] = {}
                weight_source = method
                if method == "simple_mean":
                    ensemble = values.mean(axis=1)
                elif method == "median":
                    ensemble = np.median(values, axis=1)
                elif method == "constrained_learned":
                    cache_key = (zone, target, tuple(components))
                    if cache_key not in prepared_cache:
                        prepared_cache[cache_key] = [
                            prepared for _, prior in series.groupby("run_date", sort=True)
                            if not (prepared := _complete_run(prior, components + ["actual_mw"])).empty
                        ]
                    # Run-date ordering alone does not prove delivery outcomes have finished.
                    origin = current["timestamp"].min()
                    prepared_runs = [prior for prior in prepared_cache[cache_key]
                                     if str(prior["run_date"].iloc[0]) < run_date
                                     and prior["timestamp"].max() + pd.Timedelta(hours=1) <= origin]
                    window = int(settings["training_window_runs"])
                    minimum = int(settings["minimum_training_runs"])
                    bands = settings["horizon_bands"]
                    current["horizon_band"] = current["horizon"].map(lambda value: _band_for_horizon(value, bands))
                    if len(prepared_runs) < minimum:
                        fallback_by_band = {
                            band_index: _offline_fallback_weights(
                                offline_weights,
                                zone=zone,
                                target=target,
                                band=band,
                                components=components,
                                include_tso=include_tso,
                                model_columns=model_columns,
                            )
                            for band_index, band in enumerate(bands)
                        }
                        if any(value is None for value in fallback_by_band.values()):
                            # Before enough prior realized runs exist, use a disclosed
                            # equal-weight warmup. Never train with future outcomes.
                            fallback_by_band = {
                                index: np.full(len(components), 1.0 / len(components))
                                for index in range(len(bands))
                            }
                            weight_source = "equal_weight_warmup"
                        else:
                            weight_source = "offline_weight_fallback"
                        ensemble = np.zeros(24, dtype=float)
                        for band_index, band_rows in current.groupby("horizon_band"):
                            weights = fallback_by_band[int(band_index)]
                            positions = current.index.get_indexer(band_rows.index)
                            ensemble[positions] = band_rows[components].to_numpy(float) @ weights
                            weights_by_band[int(band_index)] = weights
                    else:
                        weight_source = "rolling_prior_actuals"
                        training = pd.concat(prepared_runs[-window:], ignore_index=True)
                        training["horizon_band"] = training["horizon"].map(lambda value: _band_for_horizon(value, bands))
                        ensemble = np.zeros(24, dtype=float)
                        for band_index, band_rows in current.groupby("horizon_band"):
                            band_training = training[training["horizon_band"].eq(band_index)]
                            weights = _fit_weights(
                                band_training[components].to_numpy(float),
                                band_training["actual_mw"].to_numpy(float),
                                regularization=float(settings["regularization"]),
                                maximum_weight=float(settings["maximum_component_weight"]),
                            )
                            positions = current.index.get_indexer(band_rows.index)
                            ensemble[positions] = band_rows[components].to_numpy(float) @ weights
                            weights_by_band[int(band_index)] = weights
                else:
                    raise ValueError(f"Unknown ensemble method: {method}")

                component_labels = [model_columns[column] for column in components]
                weight_note = ""
                if weights_by_band:
                    weight_note = "|".join(
                        f"{band}:{','.join(f'{label}={weight:.4f}' for label, weight in zip(component_labels, weights))}"
                        for band, weights in sorted(weights_by_band.items())
                    )
                run_forecasts = forecast_runs.get((run_date, zone, target), forecasts.iloc[:0])
                reference = run_forecasts[run_forecasts["model"].isin(components)].copy()
                reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True)
                reference = reference.sort_values("run_at").drop_duplicates("timestamp", keep="last").set_index("timestamp")
                result = pd.DataFrame(
                    {
                        "run_date": run_date,
                        "run_at": generated_at,
                        "source": (
                            "historical_mock_ensemble"
                            if source == "online" and only_missing
                            and run_date not in (refresh_dates or set())
                            else source
                        ),
                        "zone": zone,
                        "target": target,
                        "model": ENSEMBLE_MODEL_KEY,
                        "model_label": "Ensembled model",
                        "covariate_case": f"{weight_source}:{'+'.join(component_labels)}",
                        "ensemble_weights": weight_note,
                        "ensemble_components": component_signature,
                        "context_hours": int(reference["context_hours"].max()),
                        "timestamp": current["timestamp"].to_numpy(),
                        "forecast_mw": np.maximum(ensemble, 0.0),
                        "tso_forecast_mw": current.get("tso_reference", pd.Series(np.nan,index=current.index)).to_numpy(float),
                        "actual_mw": current["actual_mw"].to_numpy(),
                        "context_start": reference["context_start"].min(),
                        "context_end": reference["context_end"].max(),
                    }
                )
                outputs.append(result)

    if not outputs:
        return pd.DataFrame()
    combined = pd.concat(outputs, ignore_index=True)
    for run_date, frame in combined.groupby("run_date"):
        store.append_forecasts(frame, str(run_date), replace_run=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the selected online ensemble forecasts.")
    parser.add_argument("--actuals", type=Path, required=True)
    parser.add_argument("--date", action="append", default=[])
    parser.add_argument("--end")
    parser.add_argument("--recent-days", type=int, default=1)
    parser.add_argument("--source", default="online")
    parser.add_argument("--repair-history", action="store_true",
                        help="Also fill missing ensembles throughout retained 90-day history.")
    args = parser.parse_args()
    if args.date:
        run_dates = args.date
    elif args.end:
        end = pd.Timestamp(args.end).date()
        start = end - timedelta(days=max(args.recent_days - 1, 0))
        run_dates = [day.date().isoformat() for day in pd.date_range(start, end, freq="D")]
    else:
        raise SystemExit("Provide --date or --end.")
    refresh_dates = set(run_dates)
    store = ForecastStore()
    if args.repair_history and not args.date:
        history = store.read_forecasts()
        latest = pd.Timestamp(args.end or history["run_date"].max())
        first = latest - pd.Timedelta(days=89)
        run_dates = sorted(set(run_dates) | set(history.loc[
            pd.to_datetime(history["run_date"]).between(first, latest), "run_date"
        ].astype(str)))
    result = build_selected_ensembles(
        store,
        actuals_path=args.actuals,
        run_dates=run_dates,
        source=args.source,
        only_missing=args.repair_history,
        refresh_dates=refresh_dates,
    )
    print(f"Wrote {len(result)} selected ensemble rows.")


if __name__ == "__main__":
    main()
