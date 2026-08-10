from __future__ import annotations

import argparse
from datetime import UTC, date, datetime

import pandas as pd

from rt_forecast_dashboard.config import features, zones
from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.data.entsoe_client import EntsoeForecastClient
from rt_forecast_dashboard.data.weather_client import OpenMeteoClient
from rt_forecast_dashboard.features import build_feature_frame
from rt_forecast_dashboard.models.registry import iter_model_adapters
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import latest_complete_run_date
from rt_forecast_dashboard.weather_points import selected_weather_points


def run_daily_forecast(run_date: date | None = None, model_keys: set[str] | None = None) -> pd.DataFrame:
    run_date = run_date or latest_complete_run_date()
    run_date_str = run_date.isoformat()
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    store = ForecastStore()
    zone_config = zones()
    target_config = features()
    entsoe = EntsoeForecastClient()
    weather_client = OpenMeteoClient()
    outputs: list[pd.DataFrame] = []
    store.clear_issues_for_run(run_date_str)

    for zone, zcfg in zone_config.items():
        for target, fcfg in target_config.items():
            try:
                horizon = int(fcfg["horizon_hours"])
                context_hours = int(fcfg["context_hours"])
                point_type = fcfg["weather_point_type"]
                points = selected_weather_points(point_type, zone)
                tso = entsoe.fetch_tso_forecast(
                    run_date=run_date,
                    zone=zone,
                    zone_code=zcfg["entsoe_code"],
                    target=target,
                    horizon_hours=horizon,
                )
                actual_context = entsoe.fetch_actual_context(
                    run_date=run_date,
                    zone=zone,
                    zone_code=zcfg["entsoe_code"],
                    target=target,
                    context_hours=context_hours,
                )
                feature_frame = build_feature_frame(tso, tso[["timestamp"]]).head(horizon)
                context_frame = build_feature_frame(actual_context, actual_context[["timestamp"]]).dropna(subset=["actual_mw"]).tail(context_hours)
                weather_loaded = False

                def ensure_weather_features() -> None:
                    nonlocal feature_frame, context_frame, weather_loaded
                    if weather_loaded:
                        return
                    weather = weather_client.fetch_future_weather(
                        run_date=run_date,
                        zone=zone,
                        points=points,
                        target=target,
                        horizon_hours=horizon,
                    )
                    weather_context = weather_client.fetch_context_weather(
                        run_date=run_date,
                        zone=zone,
                        points=points,
                        target=target,
                        context_hours=context_hours + 72,
                    )
                    feature_frame = build_feature_frame(tso, weather).head(horizon)
                    context_timestamps = actual_context[["timestamp"]].drop_duplicates()
                    weather_context = context_timestamps.merge(weather_context, on="timestamp", how="left")
                    context_frame = build_feature_frame(actual_context, weather_context).dropna(subset=["actual_mw"]).tail(context_hours)
                    weather_loaded = True

                for model_key, model_config, adapter in iter_model_adapters(target):
                    if model_keys is not None and model_key not in model_keys:
                        continue
                    try:
                        covariate_case, covariates = covariates_for(target, zone, model_key)
                        missing_covariates = [column for column in covariates if column not in feature_frame.columns or column not in context_frame.columns]
                        if missing_covariates:
                            ensure_weather_features()
                        forecast = adapter.predict(feature_frame, target, context=context_frame, covariates=covariates)
                        outputs.append(
                            pd.DataFrame(
                                {
                                    "run_date": run_date_str,
                                    "run_at": run_at,
                                    "source": "online",
                                    "zone": zone,
                                    "target": target,
                                    "model": model_key,
                                    "model_label": model_config.get("label", model_key),
                                    "covariate_case": covariate_case,
                                    "context_hours": context_hours,
                                    "timestamp": feature_frame["timestamp"],
                                    "forecast_mw": forecast,
                                    "tso_forecast_mw": feature_frame["tso_forecast_mw"],
                                    "actual_mw": pd.NA,
                                    "context_start": context_frame["timestamp"].min(),
                                    "context_end": context_frame["timestamp"].max(),
                                }
                            )
                        )
                    except Exception as exc:
                        store.log_issue(
                            run_date=run_date_str,
                            zone=zone,
                            target=target,
                            stage=f"model:{model_key}",
                            message=str(exc),
                        )
                store.write_raw(feature_frame, run_date_str, zone, target)
            except Exception as exc:
                store.log_issue(
                    run_date=run_date_str,
                    zone=zone,
                    target=target,
                    stage="data_or_features",
                    message=str(exc),
                )

    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if not result.empty:
        store.append_forecasts(result, run_date_str)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily real-time forecast pipeline.")
    parser.add_argument("--date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_keys = {value.strip() for value in args.models.split(",")} if args.models else None
    result = run_daily_forecast(args.date, model_keys=model_keys)
    print(f"Wrote {len(result)} forecast rows.")


if __name__ == "__main__":
    main()
