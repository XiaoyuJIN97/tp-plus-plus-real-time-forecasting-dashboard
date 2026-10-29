from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import sys

import numpy as np
import pandas as pd

from rt_forecast_dashboard.config import features, zones, model_supports_scope
from rt_forecast_dashboard.context_quality import ContextDataQualityError, repair_context
from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.data.entsoe_client import EntsoeForecastClient
from rt_forecast_dashboard.data.weather_client import OpenMeteoClient
from rt_forecast_dashboard.features import build_feature_frame
from rt_forecast_dashboard.models.registry import iter_model_adapters
from rt_forecast_dashboard.pipeline.check_forecast_complete import complete_forecast_groups
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import latest_complete_run_date, forecast_horizon_hours
from rt_forecast_dashboard.weather_points import selected_weather_points


def run_daily_forecast(run_date: date | None = None, model_keys: set[str] | None = None, *, clear_issues: bool = True, only_missing: bool = False, scopes: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    run_date = run_date or latest_complete_run_date()
    run_date_str = run_date.isoformat()
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    store = ForecastStore()
    zone_config = zones()
    target_config = features()
    entsoe = EntsoeForecastClient()
    weather_client = OpenMeteoClient()
    outputs: list[pd.DataFrame] = []
    completed = complete_forecast_groups(store.forecast_path(run_date_str)) if only_missing else set()
    if clear_issues:
        store.clear_issues_for_run(run_date_str)

    for zone, zcfg in zone_config.items():
        enabled_targets = set(zcfg.get("targets", target_config))
        for target, fcfg in target_config.items():
            if target not in enabled_targets or (scopes is not None and (zone, target) not in scopes):
                continue
            try:
                pending = [(key,config,adapter) for key,config,adapter in iter_model_adapters(target)
                           if model_supports_scope(config,zone,target)
                           and (model_keys is None or key in model_keys)
                           and (zone,target,key) not in completed]
                if not pending:
                    continue
                horizon = forecast_horizon_hours(run_date, int(fcfg["horizon_hours"]))
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
                expected_context_end = pd.to_datetime(tso["timestamp"], utc=True).min() - pd.Timedelta(hours=1)
                try:
                    actual_context, context_quality = repair_context(
                        actual_context,
                        expected_end=expected_context_end,
                        context_hours=context_hours,
                    )
                except ContextDataQualityError as archive_exc:
                    direct_context = entsoe.fetch_direct_actual_context(
                        run_date=run_date,
                        zone_code=zcfg["entsoe_code"],
                        target=target,
                        context_hours=context_hours,
                    )
                    if direct_context.empty:
                        raise archive_exc
                    combined_context = pd.concat([actual_context, direct_context], ignore_index=True)
                    combined_context["timestamp"] = pd.to_datetime(combined_context["timestamp"], utc=True)
                    combined_context = combined_context.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                    actual_context, context_quality = repair_context(
                        combined_context,
                        expected_end=expected_context_end,
                        context_hours=context_hours,
                    )
                    store.log_issue(
                        run_date=run_date_str,
                        zone=zone,
                        target=target,
                        stage="context_direct_fetch",
                        message=(
                            "Recovered missing archive context with direct ENTSO-E fetch after "
                            f"archive repair failed: {archive_exc}"
                        ),
                        status="recovered",
                    )
                if context_quality.imputed_hours:
                    store.log_issue(
                        run_date=run_date_str,
                        zone=zone,
                        target=target,
                        stage="context_imputation",
                        message=(
                            f"Recovered {context_quality.imputed_hours} missing context hour(s) "
                            f"using {context_quality.methods}."
                        ),
                        context={"records": list(context_quality.records)},
                        status="recovered",
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
                    weather = weather.copy()
                    weather['timestamp'] = pd.to_datetime(weather.timestamp, utc=True)
                    weather = weather.drop_duplicates('timestamp', keep='last').set_index('timestamp')
                    weather = weather.reindex(pd.DatetimeIndex(tso.timestamp))
                    if weather.isna().any().any():
                        raise ValueError('Weather does not cover every delivery hour through next-day midnight')
                    feature_frame = build_feature_frame(tso, weather.reset_index()).head(horizon)
                    context_timestamps = actual_context[["timestamp"]].drop_duplicates()
                    weather_context = context_timestamps.merge(weather_context, on="timestamp", how="left")
                    context_frame = build_feature_frame(actual_context, weather_context).dropna(subset=["actual_mw"]).tail(context_hours)
                    weather_loaded = True

                for model_key, model_config, adapter in pending:
                    if not model_supports_scope(model_config, zone, target):
                        continue
                    if model_keys is not None and model_key not in model_keys:
                        continue
                    try:
                        covariate_case, covariates = covariates_for(target, zone, model_key)
                        missing_covariates = [column for column in covariates if column not in feature_frame.columns or column not in context_frame.columns]
                        if missing_covariates:
                            ensure_weather_features()
                        forecast = np.asarray(adapter.predict(feature_frame, target, context=context_frame, covariates=covariates), dtype=float)
                        if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
                            raise ValueError(f'Expected {horizon} finite predictions through next-day midnight')
                        outputs.append(
                            pd.DataFrame(
                                {
                                    "run_date": run_date_str,
                                    "run_at": run_at,
                                    "source": "online",
                                    "forecast_window": "next_day_midnight_v1",
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
                                    "context_missing_hours": context_quality.missing_hours,
                                    "context_imputed_hours": context_quality.imputed_hours,
                                    "context_imputation_method": context_quality.methods,
                                    "data_quality": context_quality.quality,
                                }
                            )
                        )
                    except Exception as exc:
                        print(f"Issue {run_date_str} {zone} {target} model:{model_key}: {exc}", file=sys.stderr)
                        store.log_issue(
                            run_date=run_date_str,
                            zone=zone,
                            target=target,
                            stage=f"model:{model_key}",
                            message=str(exc),
                        )
                store.write_raw(feature_frame, run_date_str, zone, target)
            except Exception as exc:
                print(f"Issue {run_date_str} {zone} {target} data_or_features: {exc}", file=sys.stderr)
                store.log_issue(
                    run_date=run_date_str,
                    zone=zone,
                    target=target,
                    stage="data_or_features",
                    message=str(exc),
                )

    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if not result.empty:
        store.append_forecasts(result, run_date_str, replace_run=model_keys is None and not only_missing)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily real-time forecast pipeline.")
    parser.add_argument("--date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys to run.")
    parser.add_argument("--keep-issues", action="store_true", help="Append issues instead of clearing existing issues for the run date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_keys = {value.strip() for value in args.models.split(",")} if args.models else None
    result = run_daily_forecast(args.date, model_keys=model_keys, clear_issues=not args.keep_issues)
    print(f"Wrote {len(result)} forecast rows.")


if __name__ == "__main__":
    main()
