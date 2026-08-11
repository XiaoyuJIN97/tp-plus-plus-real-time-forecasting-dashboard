from __future__ import annotations

from datetime import date

import pandas as pd

from rt_forecast_dashboard.data.demo import demo_context_frame, demo_future_frame
from rt_forecast_dashboard.features import build_feature_frame
from rt_forecast_dashboard.models.adapters import RidgeContextAdapter, TsoReferenceAdapter
from rt_forecast_dashboard.storage import ForecastStore


def test_demo_series_has_expected_horizon() -> None:
    frame = demo_future_frame(date(2026, 8, 7), "BE", "load", 24)
    assert len(frame) == 24
    assert {"timestamp", "tso_forecast_mw", "temperature_2m_p1", "relative_humidity_2m_p1"}.issubset(frame.columns)


def test_feature_frame_and_baselines_predict() -> None:
    demo = demo_future_frame(date(2026, 8, 7), "FR", "solar", 24)
    context = demo_context_frame(date(2026, 8, 7), "FR", "solar", 24 * 92)
    features = build_feature_frame(demo[["timestamp", "tso_forecast_mw"]], demo.drop(columns=["tso_forecast_mw"]))
    context_features = build_feature_frame(
        context[["timestamp", "actual_mw", "tso_forecast_mw"]],
        context.drop(columns=["actual_mw", "tso_forecast_mw"]),
    )
    tso = TsoReferenceAdapter().predict(features, "solar")
    adjusted = RidgeContextAdapter().predict(
        features,
        "solar",
        context=context_features,
        covariates=["shortwave_radiation_p1", "shortwave_radiation_p2", "temperature_2m_p1", "tso_forecast_mw"],
    )
    assert len(tso) == len(features)
    assert len(adjusted) == len(features)
    assert adjusted.min() >= 0


def test_full_rerun_replaces_stale_model_rows(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path)
    old = pd.DataFrame(
        {
            "run_date": ["2026-08-09", "2026-08-09"],
            "run_at": ["2026-08-10T00:00:00+00:00", "2026-08-10T00:00:00+00:00"],
            "source": ["online", "online"],
            "zone": ["BE", "BE"],
            "target": ["solar", "solar"],
            "model": ["tabpfn_online", "tso_reference"],
            "model_label": ["TabPFN with covariates", "TSO forecast"],
            "covariate_case": ["weather_4p+tso_forecast", "tso_forecast"],
            "context_hours": [2208, 2208],
            "timestamp": pd.to_datetime(["2026-08-09T18:00:00Z", "2026-08-09T18:00:00Z"]),
            "forecast_mw": [1.0, 2.0],
            "tso_forecast_mw": [2.0, 2.0],
            "actual_mw": [pd.NA, pd.NA],
            "context_start": ["2026-05-07T20:00:00Z", "2026-05-09T16:00:00Z"],
            "context_end": ["2026-08-07T19:00:00Z", "2026-08-09T15:00:00Z"],
        }
    )
    store.append_forecasts(old, "2026-08-09")

    fresh = old[old["model"].eq("tso_reference")].copy()
    fresh["timestamp"] = pd.to_datetime(["2026-08-09T16:00:00Z"])
    fresh["forecast_mw"] = [3.0]
    store.append_forecasts(fresh, "2026-08-09", replace_run=True)

    saved = pd.read_csv(store.forecast_path("2026-08-09"))
    assert saved["model"].tolist() == ["tso_reference"]
    assert saved["forecast_mw"].tolist() == [3.0]
