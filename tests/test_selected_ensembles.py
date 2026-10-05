from __future__ import annotations

import numpy as np
import pandas as pd

from rt_forecast_dashboard.pipeline.build_selected_ensembles import build_selected_ensembles
from rt_forecast_dashboard.storage import ForecastStore


def _component_rows(run_date: str, zone: str, target: str, actual: np.ndarray) -> pd.DataFrame:
    timestamps = pd.date_range(f"{run_date}T00:00:00Z", periods=24, freq="h")
    ml_model = "ridge_3mo_context" if target == "load" else "xgboost_online"
    offsets = {
        "chronos2_online": 10.0,
        ml_model: -5.0,
        "timesfm3_online": 2.0,
        "tso_reference": 20.0,
    }
    rows = []
    for model, offset in offsets.items():
        rows.append(
            pd.DataFrame(
                {
                    "run_date": run_date,
                    "run_at": f"{run_date}T18:30:00Z",
                    "source": "test",
                    "zone": zone,
                    "target": target,
                    "model": model,
                    "model_label": model,
                    "covariate_case": "test",
                    "context_hours": 100,
                    "timestamp": timestamps,
                    "forecast_mw": actual + offset,
                    "tso_forecast_mw": actual + 20.0,
                    "actual_mw": pd.NA,
                    "context_start": timestamps.min() - pd.Timedelta(days=7),
                    "context_end": timestamps.min() - pd.Timedelta(hours=1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_build_selected_ensembles_uses_offline_selected_configuration(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path / "forecast-data")
    actual_parts = []
    run_dates = pd.date_range("2026-01-01", periods=22, freq="D")
    for day in run_dates:
        run_date = day.date().isoformat()
        actual = np.linspace(100.0, 200.0, 24)
        store.append_forecasts(_component_rows(run_date, "BE", "load", actual), run_date)
        timestamps = pd.date_range(f"{run_date}T00:00:00Z", periods=24, freq="h")
        actual_parts.append(
            pd.DataFrame(
                {
                    "timestamp_utc": timestamps,
                    "collection_time_utc": timestamps + pd.Timedelta(hours=2),
                    "zone": "BE",
                    "target": "load",
                    "actual_mw": actual,
                }
            )
        )
    actuals_path = tmp_path / "actuals.parquet"
    pd.concat(actual_parts, ignore_index=True).to_parquet(actuals_path, index=False)

    result = build_selected_ensembles(
        store,
        actuals_path=actuals_path,
        run_dates=[run_dates[-1].date().isoformat()],
    )

    assert len(result) == 24
    assert result["model"].unique().tolist() == ["ensemble_selected"]
    assert result["model_label"].unique().tolist() == [
        "Ensemble — Constrained learned (without TSO)"
    ]
    assert "TSO" not in result["covariate_case"].iloc[0].split(";", 1)[0]
    stored = store.read_forecasts()
    assert len(stored[stored["model"].eq("ensemble_selected")]) == 24


def test_simple_mean_selected_configuration(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path / "forecast-data")
    run_date = "2026-01-22"
    actual = np.linspace(100.0, 200.0, 24)
    store.append_forecasts(_component_rows(run_date, "DE", "wind_offshore", actual), run_date)
    timestamps = pd.date_range(f"{run_date}T00:00:00Z", periods=24, freq="h")
    actuals_path = tmp_path / "actuals.parquet"
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "collection_time_utc": timestamps + pd.Timedelta(hours=2),
            "zone": "DE",
            "target": "wind_offshore",
            "actual_mw": actual,
        }
    ).to_parquet(actuals_path, index=False)

    result = build_selected_ensembles(store, actuals_path=actuals_path, run_dates=[run_date])

    assert len(result) == 24
    assert result["model_label"].unique().tolist() == ["Ensemble — Simple mean (without TSO)"]
    np.testing.assert_allclose(result["forecast_mw"], actual + (10.0 - 5.0 + 2.0) / 3.0)
