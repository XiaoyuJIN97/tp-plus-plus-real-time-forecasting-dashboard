from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rt_forecast_dashboard.config import load_yaml
from rt_forecast_dashboard.storage import ForecastStore


ENSEMBLE_MODEL_KEY = "ensemble_selected"
MODEL_COLUMNS = {
    "chronos2_online": "Chronos2",
    "ridge_3mo_context": "Ridge",
    "xgboost_online": "XGBoost",
    "timesfm3_online": "TimesFM3",
    "tso_reference": "TSO",
}
METHOD_LABELS = {
    "simple_mean": "Simple mean",
    "median": "Median",
    "constrained_learned": "Constrained learned",
}


def _component_columns(target: str, include_tso: bool) -> list[str]:
    ml_model = "ridge_3mo_context" if target == "load" else "xgboost_online"
    columns = ["chronos2_online", ml_model, "timesfm3_online"]
    if include_tso:
        columns.append("tso_reference")
    return columns


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


def _forecast_panel(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    components = set(MODEL_COLUMNS)
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
    if len(complete) != 24 or complete["timestamp"].nunique() != 24:
        return pd.DataFrame()
    complete["horizon"] = np.arange(1, 25)
    return complete


def _band_for_horizon(horizon: int, bands: list[list[int]]) -> int:
    for index, (start, end) in enumerate(bands):
        if start <= horizon <= end:
            return index
    raise ValueError(f"Horizon {horizon} is not covered by the configured bands.")


def build_selected_ensembles(
    store: ForecastStore,
    *,
    actuals_path: Path,
    run_dates: list[str],
) -> pd.DataFrame:
    config = load_yaml("ensemble_registry.yml")
    settings = config["settings"]
    selections = config["selections"]
    forecasts = store.read_forecasts()
    actuals = _load_actuals(actuals_path)
    panel = _forecast_panel(forecasts, actuals)
    outputs: list[pd.DataFrame] = []
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    for run_date in sorted(set(run_dates)):
        for zone, target_configs in selections.items():
            for target, selection in target_configs.items():
                method = selection["method"]
                include_tso = bool(selection["include_tso"])
                components = _component_columns(target, include_tso)
                current = _complete_run(
                    panel[
                        panel["run_date"].astype(str).eq(run_date)
                        & panel["zone"].eq(zone)
                        & panel["target"].eq(target)
                    ],
                    components,
                )
                if current.empty:
                    print(f"Skip ensemble {run_date} {zone} {target}: component forecasts are incomplete.")
                    continue

                values = current[components].to_numpy(float)
                weights_by_band: dict[int, np.ndarray] = {}
                if method == "simple_mean":
                    ensemble = values.mean(axis=1)
                elif method == "median":
                    ensemble = np.median(values, axis=1)
                elif method == "constrained_learned":
                    earlier = panel[
                        panel["run_date"].astype(str).lt(run_date)
                        & panel["zone"].eq(zone)
                        & panel["target"].eq(target)
                    ].copy()
                    complete_dates: list[str] = []
                    prepared_runs: list[pd.DataFrame] = []
                    for prior_date, prior in earlier.groupby("run_date"):
                        prepared = _complete_run(prior, components + ["actual_mw"])
                        if prepared.empty:
                            continue
                        complete_dates.append(str(prior_date))
                        prepared_runs.append(prepared)
                    window = int(settings["training_window_runs"])
                    minimum = int(settings["minimum_training_runs"])
                    if len(prepared_runs) < minimum:
                        print(
                            f"Skip ensemble {run_date} {zone} {target}: "
                            f"{len(prepared_runs)} complete training runs, need {minimum}."
                        )
                        continue
                    training = pd.concat(prepared_runs[-window:], ignore_index=True)
                    bands = settings["horizon_bands"]
                    training["horizon_band"] = training["horizon"].map(lambda value: _band_for_horizon(value, bands))
                    current["horizon_band"] = current["horizon"].map(lambda value: _band_for_horizon(value, bands))
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

                tso_option = "with TSO" if include_tso else "without TSO"
                component_labels = [MODEL_COLUMNS[column] for column in components]
                weight_note = ""
                if weights_by_band:
                    weight_note = "; weights=" + "|".join(
                        f"{band}:{','.join(f'{label}={weight:.4f}' for label, weight in zip(component_labels, weights))}"
                        for band, weights in sorted(weights_by_band.items())
                    )
                reference = forecasts[
                    forecasts["run_date"].astype(str).eq(run_date)
                    & forecasts["zone"].eq(zone)
                    & forecasts["target"].eq(target)
                    & forecasts["model"].isin(components)
                ].copy()
                reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True)
                reference = reference.sort_values("run_at").drop_duplicates("timestamp", keep="last").set_index("timestamp")
                result = pd.DataFrame(
                    {
                        "run_date": run_date,
                        "run_at": generated_at,
                        "source": "online",
                        "zone": zone,
                        "target": target,
                        "model": ENSEMBLE_MODEL_KEY,
                        "model_label": f"Ensemble — {METHOD_LABELS[method]} ({tso_option})",
                        "covariate_case": f"offline_selected:{'+'.join(component_labels)}{weight_note}",
                        "context_hours": int(reference["context_hours"].max()),
                        "timestamp": current["timestamp"].to_numpy(),
                        "forecast_mw": np.maximum(ensemble, 0.0),
                        "tso_forecast_mw": current["tso_reference"].to_numpy(float),
                        "actual_mw": pd.NA,
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
    args = parser.parse_args()
    if args.date:
        run_dates = args.date
    elif args.end:
        end = pd.Timestamp(args.end).date()
        start = end - timedelta(days=max(args.recent_days - 1, 0))
        run_dates = [day.date().isoformat() for day in pd.date_range(start, end, freq="D")]
    else:
        raise SystemExit("Provide --date or --end.")
    result = build_selected_ensembles(ForecastStore(), actuals_path=args.actuals, run_dates=run_dates)
    print(f"Wrote {len(result)} selected ensemble rows.")


if __name__ == "__main__":
    main()
