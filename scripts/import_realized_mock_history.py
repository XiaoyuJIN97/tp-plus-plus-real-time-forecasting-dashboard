from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rt_forecast_dashboard.storage import ForecastStore


TP_ROOT = Path("/Users/xiaoyujin/Desktop/TP++")
SOURCE = "historical_mock_realized"
CONTEXT_HOURS = 2208


def _delivery_timestamp(run_date: str, horizon_index: pd.Series) -> pd.Series:
    start = pd.to_datetime(run_date, utc=True) + pd.Timedelta(hours=18)
    return start + pd.to_timedelta(horizon_index.astype(int), unit="h")


def _standard_rows(
    frame: pd.DataFrame,
    *,
    target: str,
    model_key: str,
    model_label: str,
    covariate_case: str,
    forecast_col: str,
    horizon_index: pd.Series,
) -> pd.DataFrame:
    run_dates = pd.to_datetime(frame["cutoff"], utc=True).dt.date.astype(str)
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "run_date": run_dates,
            "run_at": pd.to_datetime(frame["cutoff"], utc=True),
            "zone": frame["country"].astype(str),
            "target": target,
            "model": model_key,
            "model_label": model_label,
            "covariate_case": covariate_case,
            "context_hours": CONTEXT_HOURS,
            "timestamp": pd.NaT,
            "forecast_mw": pd.to_numeric(frame[forecast_col], errors="coerce"),
            "actual_mw": pd.to_numeric(frame["y_true"], errors="coerce"),
            "tso_forecast_mw": pd.NA,
            "context_start": pd.to_datetime(frame["cutoff"], utc=True) - pd.Timedelta(hours=CONTEXT_HOURS),
            "context_end": pd.to_datetime(frame["cutoff"], utc=True) - pd.Timedelta(hours=1),
        }
    )
    out["timestamp"] = [
        _delivery_timestamp(run_date, pd.Series([idx])).iloc[0]
        for run_date, idx in zip(out["run_date"], horizon_index, strict=False)
    ]
    return out.dropna(subset=["forecast_mw", "actual_mw"])


def _load_rows(start: date, end: date) -> pd.DataFrame:
    path = TP_ROOT / "Load_forecast_new/load_forecast_outputs_daily_18utc_2024_2025_4p_weather/csv/selected_model_variant_forecasts.csv"
    frame = pd.read_csv(path)
    frame["cutoff_date"] = pd.to_datetime(frame["cutoff"], utc=True).dt.date
    frame = frame[frame["cutoff_date"].between(start, end)].copy()
    selected = [
        ("TSO Forecast", "tso_reference", "TSO forecast", "tso_forecast"),
        ("Weekly Persistence", "persistence", "Persistence", "weekly_persistence"),
        ("Ridge", "ridge_3mo_context", "Ridge with covariates", "selected_covariates+tso_forecast"),
    ]
    rows = []
    for base_model, model_key, label, case in selected:
        part = frame[frame["base_model"].eq(base_model)].copy()
        rows.append(
            _standard_rows(
                part,
                target="load",
                model_key=model_key,
                model_label=label,
                covariate_case=case,
                forecast_col="prediction",
                horizon_index=pd.to_numeric(part["horizon"], errors="coerce").fillna(0).astype(int),
            )
        )
    return pd.concat(rows, ignore_index=True)


def _solar_rows(start: date, end: date) -> pd.DataFrame:
    folder = TP_ROOT / "Solar_forecast_tabpfn_new/solar_weather_tso_cov_outputs/csv"
    selected = {
        "BE": "XGBoost_Weather_TSOCov",
        "FR": "XGBoost_TSOCov",
        "DE": "XGBoost_Weather_TSOCov",
    }
    rows = []
    for country in ["BE", "FR", "DE"]:
        files = [
            ("TSO_Forecast", "tso_reference", "TSO forecast", "tso_forecast"),
            ("Daily_Persistence", "persistence", "Persistence", "daily_persistence"),
            (selected[country], "xgboost_online", "XGBoost with covariates", selected[country]),
        ]
        for file_key, model_key, label, case in files:
            path = folder / f"{country}_Solar_{file_key}_results_eval.csv"
            part = pd.read_csv(path)
            part["cutoff_date"] = pd.to_datetime(part["cutoff"], utc=True).dt.date
            part = part[part["cutoff_date"].between(start, end)].copy()
            rows.append(
                _standard_rows(
                    part,
                    target="solar",
                    model_key=model_key,
                    model_label=label,
                    covariate_case=case,
                    forecast_col="median",
                    horizon_index=pd.to_numeric(part["horizon_hours"], errors="coerce").fillna(1).astype(int) - 1,
                )
            )
    return pd.concat(rows, ignore_index=True)


def _wind_rows(start: date, end: date, *, target: str, filename: str) -> pd.DataFrame:
    path = TP_ROOT / f"Wind_forecast_new/outputs/latest_selected_wind_comparisons/{filename}"
    frame = pd.read_csv(path)
    frame["cutoff_date"] = pd.to_datetime(frame["cutoff"], utc=True).dt.date
    frame = frame[frame["cutoff_date"].between(start, end)].copy()
    selected = [
        ("TSO", "tso_reference", "TSO forecast", "tso_forecast"),
        ("Daily_Persistence", "persistence", "Persistence", "daily_persistence"),
        ("XGBoost", "xgboost_online", "XGBoost with covariates", "weather_4p+tso_forecast"),
    ]
    rows = []
    for family, model_key, label, case in selected:
        part = frame[frame["model_family"].eq(family)].copy()
        rows.append(
            _standard_rows(
                part,
                target=target,
                model_key=model_key,
                model_label=label,
                covariate_case=case,
                forecast_col="median",
                horizon_index=pd.to_numeric(part["horizon_hours"], errors="coerce").fillna(1).astype(int) - 1,
            )
        )
    return pd.concat(rows, ignore_index=True)


def import_realized_mock_history(days: int = 14) -> pd.DataFrame:
    end = date(2025, 12, 30)
    start = end - timedelta(days=days - 1)
    frame = pd.concat(
        [
            _load_rows(start, end),
            _solar_rows(start, end),
            _wind_rows(start, end, target="wind_onshore", filename="onshore_five_case_eval_df.csv"),
            _wind_rows(start, end, target="wind_offshore", filename="offshore_five_case_eval_df.csv"),
        ],
        ignore_index=True,
    )
    store = ForecastStore()
    for run_date, part in frame.groupby("run_date"):
        store.append_forecasts(part, str(run_date))
    return frame


if __name__ == "__main__":
    result = import_realized_mock_history()
    print(f"Imported {len(result):,} realized mock forecast rows.")
    print(f"Dates: {result['run_date'].min()} to {result['run_date'].max()}")
