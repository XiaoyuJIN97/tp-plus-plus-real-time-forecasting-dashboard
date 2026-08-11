from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rt_forecast_dashboard.config import model_registry
from rt_forecast_dashboard.time_utils import latest_complete_run_date


TP_ROOT = Path("/Users/xiaoyujin/Desktop/TP++")
DAY_AHEAD_HOURS = 24
ONLINE_CONTEXT_HOURS = 2208
MODEL_FAMILY_ORDER = ["Chronos2", "Ridge", "TabPFN", "XGBoost", "TSO forecast", "Persistence"]
MODEL_FAMILY_RANK = {family: rank for rank, family in enumerate(MODEL_FAMILY_ORDER)}


def _normal_model_family(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower().replace("_", " ")
    if "chronos" in lowered:
        return "Chronos2"
    if "ridge" in lowered:
        return "Ridge"
    if "tabpfn" in lowered:
        return "TabPFN"
    if "xgboost" in lowered:
        return "XGBoost"
    if lowered.startswith("tso") or "tso forecast" in lowered:
        return "TSO forecast"
    if "persistence" in lowered or lowered == "daily":
        return "Persistence"
    return text


def online_enabled_model_families(target: str) -> list[str]:
    families = []
    for model_key, config in model_registry().items():
        if not config.get("enabled", False) or target not in config.get("targets", []):
            continue
        families.append(_normal_model_family(config.get("label", model_key)))
    return [family for family in MODEL_FAMILY_ORDER if family in set(families)]


def load_historical_accuracy(target: str) -> pd.DataFrame:
    if target == "load":
        path = TP_ROOT / "Load_forecast_new/load_forecast_outputs_daily_18utc_2024_2025_4p_weather/csv/accuracy_metrics.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        frame["target"] = "load"
        frame["display_model"] = frame["model"]
        return frame[["target", "country", "display_model", "base_model", "case", "MAE", "RMSE", "MAPE", "R2", "n"]]

    if target == "solar":
        folder = TP_ROOT / "Solar_forecast_tabpfn_new/solar_weather_tso_cov_outputs/csv"
        rows = []
        for path in folder.glob("*_Solar_*_accuracy_summary.csv"):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            row = frame.iloc[0].to_dict()
            model = row["model"]
            rows.append(
                {
                    "target": "solar",
                    "country": row["country"],
                    "display_model": model,
                    "base_model": str(model).split("_", 1)[0],
                    "case": str(model).split("_", 1)[1] if "_" in str(model) else "Baseline",
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE": row["MAPE"],
                    "R2": row["R2"],
                    "n": row["n"],
                }
            )
        return pd.DataFrame(rows)

    if target in {"wind_onshore", "wind_offshore"}:
        name = "onshore" if target == "wind_onshore" else "offshore"
        path = TP_ROOT / f"Wind_forecast_new/outputs/latest_selected_wind_comparisons/{name}_six_case_accuracy_table.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        frame["target"] = target
        frame["display_model"] = frame["Selected model"]
        frame["base_model"] = frame["model_family"]
        frame["n"] = frame["n_samples"]
        rows = [frame[["target", "country", "display_model", "base_model", "case", "MAE", "RMSE", "MAPE", "R2", "n"]]]
        wind_name = "Wind_Onshore" if target == "wind_onshore" else "Wind_Offshore"
        for country in ["BE", "FR", "DE"]:
            for case_name, display_name in [
                ("XGBoost_Wind100mCovariates", "XGBoost + 100m wind"),
                ("XGBoost_NoCovariates", "XGBoost"),
            ]:
                summary_path = TP_ROOT / f"Wind_forecast_new/outputs/wind_forecast/{country}/{wind_name}/{case_name}/accuracy/summary.csv"
                if not summary_path.exists():
                    continue
                summary = pd.read_csv(summary_path)
                if summary.empty:
                    continue
                row = summary.iloc[0]
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "target": target,
                                "country": country,
                                "display_model": display_name,
                                "base_model": "XGBoost",
                                "case": case_name.replace("XGBoost_", ""),
                                "MAE": row["MAE"],
                                "RMSE": row["RMSE"],
                                "MAPE": row["MAPE"],
                                "R2": row["R2"],
                                "n": row["n_samples"],
                            }
                        ]
                    )
                )
        return pd.concat(rows, ignore_index=True)

    return pd.DataFrame()


def optimal_accuracy_cases(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["display_family"] = frame["base_model"].map(_normal_model_family)
    frame["family_rank"] = frame["display_family"].map(MODEL_FAMILY_RANK).fillna(len(MODEL_FAMILY_ORDER)).astype(int)
    metric = "RMSE" if "RMSE" in frame.columns else "MAE"
    required = {"country", "display_family", metric}
    if not required.issubset(frame.columns):
        return frame
    sort_cols = ["country", "display_family", metric]
    ascending = [True, True, metric != "R2"]
    return (
        frame.dropna(subset=[metric])
        .sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["country", "display_family"], keep="first")
        .sort_values(["country", "family_rank"])
        .reset_index(drop=True)
    )


def online_accuracy_cases(target: str) -> pd.DataFrame:
    accuracy = optimal_accuracy_cases(load_historical_accuracy(target))
    if accuracy.empty:
        return accuracy
    enabled_families = online_enabled_model_families(target)
    accuracy = accuracy[accuracy["display_family"].isin(enabled_families)].copy()
    return accuracy.sort_values(["country", "family_rank"]).reset_index(drop=True)


def online_forecast_accuracy(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    frame = forecasts.dropna(subset=["actual_mw", "forecast_mw"]).copy()
    if frame.empty:
        return pd.DataFrame()
    if {"run_date", "zone", "target", "model", "horizon"}.issubset(frame.columns):
        complete = (
            frame.groupby(["run_date", "zone", "target", "model"])["horizon"]
            .nunique()
            .reset_index(name="realized_horizon_count")
        )
        complete = complete[complete["realized_horizon_count"].eq(DAY_AHEAD_HOURS)]
        frame = frame.merge(complete[["run_date", "zone", "target", "model"]], on=["run_date", "zone", "target", "model"], how="inner")
        if frame.empty:
            return pd.DataFrame()
    rows = []
    group_cols = ["zone", "model_label", "covariate_case"]
    for (country, model_label, case), group in frame.groupby(group_cols, dropna=False):
        actual = group["actual_mw"].to_numpy(dtype=float)
        forecast = group["forecast_mw"].to_numpy(dtype=float)
        error = forecast - actual
        mask = np.abs(actual) > 1e-6
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        rows.append(
            {
                "country": country,
                "display_model": model_label,
                "display_family": _normal_model_family(model_label),
                "case": case,
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAPE": float(np.mean(np.abs(error[mask] / actual[mask])) * 100) if mask.any() else np.nan,
                "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                "n": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["family_rank"] = out["display_family"].map(MODEL_FAMILY_RANK).fillna(len(MODEL_FAMILY_ORDER)).astype(int)
    return out.sort_values(["country", "family_rank"]).reset_index(drop=True)


def load_historical_forecasts(target: str) -> pd.DataFrame:
    if target == "load":
        path = TP_ROOT / "Load_forecast_new/load_forecast_outputs_daily_18utc_2024_2025_4p_weather/csv/selected_model_variant_forecasts.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        frame = frame[frame["base_model"].isin(["Chronos2", "TSO Forecast", "Weekly Persistence"])].copy()
        return _standardize_history(
            frame,
            target=target,
            timestamp_col="Date",
            forecast_col="prediction",
            model_label_col="model",
            actual_col="y_true",
            cutoff_col="cutoff",
            family_col="base_model",
            case_col="case",
            context_hours=2208,
        )

    if target == "solar":
        folder = TP_ROOT / "Solar_forecast_tabpfn_new/solar_weather_tso_cov_outputs/csv"
        wanted = [
            ("Chronos2_Weather_TSOCov", "Chronos2 TSFM + weather + TSO"),
            ("TSO_Forecast", "TSO forecast"),
            ("Daily_Persistence", "Daily persistence"),
        ]
        frames = []
        for country in ["BE", "FR", "DE"]:
            for file_key, label in wanted:
                path = folder / f"{country}_Solar_{file_key}_results_eval.csv"
                if not path.exists():
                    continue
                part = pd.read_csv(path)
                part["model_label"] = label
                part["model_family"] = "Chronos2" if file_key.startswith("Chronos2") else file_key
                part["case"] = file_key
                frames.append(part)
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        return _standardize_history(
            frame,
            target=target,
            timestamp_col="timestamp",
            forecast_col="median",
            model_label_col="model_label",
            actual_col="y_true",
            cutoff_col="cutoff",
            family_col="model_family",
            case_col="case",
            context_hours=2208,
        )

    if target in {"wind_onshore", "wind_offshore"}:
        name = "onshore" if target == "wind_onshore" else "offshore"
        path = TP_ROOT / f"Wind_forecast_new/outputs/latest_selected_wind_comparisons/{name}_five_case_eval_df.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        keep = frame["model_family"].isin(["Chronos2", "TSO", "Daily_Persistence"])
        frame = frame[keep].copy()
        return _standardize_history(
            frame,
            target=target,
            timestamp_col="timestamp",
            forecast_col="median",
            model_label_col="model_label",
            actual_col="y_true",
            cutoff_col="cutoff",
            family_col="model_family",
            case_col="case",
            context_hours=2160,
        )

    return pd.DataFrame()


def _standardize_history(
    frame: pd.DataFrame,
    *,
    target: str,
    timestamp_col: str,
    forecast_col: str,
    model_label_col: str,
    actual_col: str,
    cutoff_col: str,
    family_col: str,
    case_col: str,
    context_hours: int,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "run_date": pd.to_datetime(frame[cutoff_col], utc=True).dt.date.astype(str),
            "run_at": pd.to_datetime(frame[cutoff_col], utc=True),
            "zone": frame["country"],
            "target": target,
            "model": frame[model_label_col].astype(str).str.lower().str.replace(" ", "_", regex=False),
            "model_label": frame[model_label_col],
            "model_family": frame[family_col],
            "covariate_case": frame[case_col].fillna("-").astype(str),
            "context_hours": context_hours,
            "timestamp": pd.to_datetime(frame[timestamp_col], utc=True),
            "forecast_mw": pd.to_numeric(frame[forecast_col], errors="coerce"),
            "actual_mw": pd.to_numeric(frame[actual_col], errors="coerce"),
        }
    )
    out["horizon"] = ((out["timestamp"] - pd.to_datetime(frame[cutoff_col], utc=True)).dt.total_seconds() / 3600).round().astype("Int64")
    return out.dropna(subset=["forecast_mw", "actual_mw"])


def latest_forecast_slice(forecasts: pd.DataFrame, target: str, countries: list[str]) -> pd.DataFrame:
    frame = valid_online_forecasts(forecasts)
    frame = frame[(frame["target"] == target) & (frame["zone"].isin(countries))].copy()
    if frame.empty:
        return frame
    latest_run = frame["run_date"].max()
    return frame[frame["run_date"] == latest_run].copy()


def valid_online_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()
    frame = forecasts.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["run_date"] = frame["run_date"].astype(str)
    frame = frame[frame["run_date"].le(latest_complete_run_date().isoformat())].copy()
    if "source" in frame.columns:
        frame = frame[frame["source"].eq("online")].copy()
    frame["context_hours"] = pd.to_numeric(frame.get("context_hours", 0), errors="coerce").fillna(0).astype(int)
    if "context_end" in frame.columns:
        frame["context_end"] = pd.to_datetime(frame["context_end"], utc=True, errors="coerce")
    enabled = model_registry()
    enabled_pairs = {
        (model_key, target)
        for model_key, config in enabled.items()
        if config.get("enabled", False)
        for target in config.get("targets", [])
    }
    if enabled_pairs and {"model", "target"}.issubset(frame.columns):
        frame = frame[frame[["model", "target"]].apply(tuple, axis=1).isin(enabled_pairs)].copy()
    frame = frame[frame["context_hours"].eq(ONLINE_CONTEXT_HOURS)].copy()
    if frame.empty:
        return frame
    group_cols = ["run_date", "zone", "target", "model"]
    frame = frame.sort_values(group_cols + ["timestamp"]).drop_duplicates(group_cols + ["timestamp"], keep="last")
    if "context_end" in frame.columns:
        coverage = (
            frame.groupby(group_cols, as_index=False)
            .agg(first_delivery=("timestamp", "min"), context_end=("context_end", "max"))
        )
        coverage = coverage[coverage["context_end"].ge(coverage["first_delivery"] - pd.Timedelta(hours=1))]
        frame = frame.merge(coverage[group_cols], on=group_cols, how="inner")
        if frame.empty:
            return frame
    frame["horizon"] = frame.groupby(group_cols).cumcount()
    frame = frame[frame["horizon"].between(0, DAY_AHEAD_HOURS - 1)].copy()
    complete = (
        frame.groupby(group_cols)["horizon"]
        .nunique()
        .reset_index(name="horizon_count")
    )
    complete = complete[complete["horizon_count"].eq(DAY_AHEAD_HOURS)]
    if complete.empty:
        return frame.iloc[0:0].copy()
    return frame.merge(complete[group_cols], on=group_cols, how="inner")


def filter_last_n_days(frame: pd.DataFrame, n_days: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    end = frame["timestamp"].max()
    start = end - pd.Timedelta(days=n_days)
    return frame[frame["timestamp"].between(start, end)].copy()
