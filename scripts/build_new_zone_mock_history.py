from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TP = Path("/Users/xiaoyujin/Desktop/TP++")
ZONES = ("NL", "DK1", "DK2", "ES", "PT")
TARGETS = {
    "NL": ("load", "solar", "wind_onshore", "wind_offshore"),
    "DK1": ("load", "solar", "wind_onshore", "wind_offshore"),
    "DK2": ("load", "solar", "wind_onshore", "wind_offshore"),
    "ES": ("load", "solar", "wind_onshore"),
    "PT": ("load", "solar", "wind_onshore", "wind_offshore"),
}


def run_date(cutoff: pd.Series) -> pd.Series:
    return pd.to_datetime(cutoff, utc=True).dt.tz_convert("Europe/Brussels").dt.date.astype(str)


def standard(frame: pd.DataFrame, zone: str, target: str, model: str, label: str,
             cutoff: str, timestamp: str, forecast: str, actual: str, case: str) -> pd.DataFrame:
    cut = pd.to_datetime(frame[cutoff], utc=True)
    out = pd.DataFrame({
        "source": "historical_mock_realized", "run_date": run_date(frame[cutoff]),
        "run_at": cut, "zone": zone, "target": target, "model": model,
        "model_label": label, "covariate_case": case, "context_hours": 2208,
        "timestamp": pd.to_datetime(frame[timestamp], utc=True),
        "forecast_mw": pd.to_numeric(frame[forecast], errors="coerce"),
        "actual_mw": pd.to_numeric(frame[actual], errors="coerce"),
        "tso_forecast_mw": pd.NA, "context_start": cut - pd.Timedelta(hours=2208),
        "context_end": cut - pd.Timedelta(hours=1),
    })
    return out.dropna(subset=["timestamp", "forecast_mw", "actual_mw"])


def load_rows() -> list[pd.DataFrame]:
    path = TP / "Load_forecast_new/load_forecast_outputs_daily_18utc_2024_2025_4p_weather_NL_DK_ES_PT/csv/selected_model_variant_forecasts_updated.csv"
    data = pd.read_csv(path)
    selections = {
        "Chronos2": ("chronos2_online", "Chronos2"), "Ridge": ("ridge_3mo_context", "Ridge"),
        "TSO Forecast": ("tso_reference", "TSO forecast"),
        "Weekly Persistence": ("persistence", "Persistence"),
    }
    rows = []
    for zone in ZONES:
        for family, (key, label) in selections.items():
            part = data[(data.country == zone) & (data.base_model == family)]
            rows.append(standard(part, zone, "load", key, label, "cutoff", "Date", "prediction", "y_true", family))
    return rows


def solar_rows() -> list[pd.DataFrame]:
    root = TP / "Solar_forecast_tabpfn_new/solar_4p_new_zone_outputs/csv"
    selections = {
        "Chronos2_Weather_TSOCov": ("chronos2_online", "Chronos2"),
        "XGBoost_Weather_TSOCov": ("xgboost_online", "XGBoost"),
        "TSO_Forecast": ("tso_reference", "TSO forecast"),
        "Daily_Persistence": ("persistence", "Persistence"),
    }
    rows = []
    for zone in ZONES:
        for variant, (key, label) in selections.items():
            part = pd.read_csv(root / f"{zone}_Solar_{variant}_results_eval.csv")
            rows.append(standard(part, zone, "solar", key, label, "cutoff", "timestamp", "median", "y_true", variant))
    return rows


def wind_rows() -> list[pd.DataFrame]:
    root = TP / "Wind_forecast_new/outputs/wind_forecast_new_countries_4points_daily18"
    selections = {
        "Chronos2_Weather_TSOForecast": ("chronos2_online", "Chronos2"),
        "XGBoost_Weather_TSOForecast": ("xgboost_online", "XGBoost"),
        "TSO": ("tso_reference", "TSO forecast"),
        "Daily_Persistence": ("persistence", "Persistence"),
    }
    rows = []
    for zone in ZONES:
        for kind in ("onshore", "offshore"):
            target = f"wind_{kind}"
            if target not in TARGETS[zone]:
                continue
            cutoff_reference = pd.read_csv(root / zone / kind / "Chronos2_Weather_TSOForecast" / "accuracy/eval_df.csv")[["timestamp", "cutoff"]].drop_duplicates("timestamp")
            for variant, (key, label) in selections.items():
                part = pd.read_csv(root / zone / kind / variant / "accuracy/eval_df.csv")
                if "cutoff" not in part:
                    part = part.merge(cutoff_reference, on="timestamp", how="left")
                rows.append(standard(part, zone, target, key, label, "cutoff", "timestamp", "median", "y_true", variant))
    return rows


def timesfm_rows(root: Path) -> list[pd.DataFrame]:
    rows = []
    for zone, targets in TARGETS.items():
        for target in targets:
            path = root / f"timesfm3_{zone}_{target}_2021_2025.csv"
            data = pd.read_csv(path)
            part = data[data.model.eq("timesfm3_offline")].copy()
            rows.append(standard(part, zone, target, "timesfm3_online", "TimesFM3", "cutoff_utc", "timestamp", "forecast_mw", "actual_mw", "offline_replay"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesfm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-30")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecasts = pd.concat(load_rows() + solar_rows() + wind_rows() + timesfm_rows(args.timesfm_root), ignore_index=True)
    forecasts = forecasts[forecasts.run_date.between(args.start, args.end)].copy()
    forecasts = forecasts.sort_values(["run_date", "zone", "target", "model", "timestamp"]).drop_duplicates(
        ["run_date", "zone", "target", "model", "timestamp"], keep="last")
    actuals = forecasts[["zone", "target", "timestamp", "actual_mw"]].dropna().drop_duplicates(
        ["zone", "target", "timestamp"], keep="last")
    actuals = actuals.rename(columns={"timestamp": "timestamp_utc"})
    actuals["collection_time_utc"] = pd.Timestamp.now(tz="UTC")
    forecasts.to_parquet(args.output_dir / "forecasts.parquet", index=False)
    actuals.to_parquet(args.output_dir / "actuals.parquet", index=False)
    forecasts.to_csv(args.output_dir / "forecasts.csv", index=False)
    coverage = forecasts.groupby(["zone", "target", "model"], as_index=False).agg(
        first_run=("run_date", "min"), last_run=("run_date", "max"), runs=("run_date", "nunique"), rows=("timestamp", "size"))
    coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
