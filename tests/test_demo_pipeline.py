from __future__ import annotations

from datetime import date

from rt_forecast_dashboard.data.demo import demo_context_frame, demo_future_frame
from rt_forecast_dashboard.features import build_feature_frame
from rt_forecast_dashboard.models.adapters import RidgeContextAdapter, TsoReferenceAdapter


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
