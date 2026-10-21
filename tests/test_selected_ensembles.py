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
    assert result["model_label"].unique().tolist() == ["Ensembled model"]
    assert "TSO" not in result["covariate_case"].iloc[0].split(";", 1)[0]
    assert result["covariate_case"].nunique() == 1
    assert result["ensemble_weights"].str.len().gt(0).all()
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

    result = build_selected_ensembles(
        store,
        actuals_path=actuals_path,
        run_dates=[run_date],
        source="historical_mock_ensemble",
    )

    assert len(result) == 24
    assert result["model_label"].unique().tolist() == ["Ensembled model"]
    assert result["source"].unique().tolist() == ["historical_mock_ensemble"]
    np.testing.assert_allclose(result["forecast_mw"], actual + (10.0 - 5.0 + 2.0) / 3.0)


def test_constrained_ensemble_uses_offline_weights_before_training_minimum(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path / "forecast-data")
    run_date = "2026-10-02"
    actual = np.linspace(100.0, 200.0, 24)
    store.append_forecasts(_component_rows(run_date, "DK2", "load", actual), run_date)
    timestamps = pd.date_range(f"{run_date}T00:00:00Z", periods=24, freq="h")
    actuals_path = tmp_path / "actuals.parquet"
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "zone": "DK2",
            "target": "load",
            "actual_mw": actual,
        }
    ).to_parquet(actuals_path, index=False)

    result = build_selected_ensembles(store, actuals_path=actuals_path, run_dates=[run_date])

    assert len(result) == 24
    assert result["covariate_case"].str.startswith("offline_weight_fallback:").all()
    assert result["ensemble_weights"].str.len().gt(0).all()


def test_history_repair_preserves_saved_ensembles_but_refreshes_recent_dates(tmp_path) -> None:
    store = ForecastStore(data_dir=tmp_path / "data")
    day = "2026-07-24"
    actual = np.linspace(100.0, 200.0, 24)
    store.append_forecasts(_component_rows(day, "DE", "wind_offshore", actual), day)
    path = tmp_path / "actuals.parquet"
    pd.DataFrame({"timestamp_utc": pd.date_range(day, periods=24, freq="h", tz="UTC"),
                  "zone": "DE", "target": "wind_offshore", "actual_mw": actual}).to_parquet(path)
    first = build_selected_ensembles(store, actuals_path=path, run_dates=[day])
    assert len(first) == 24
    assert build_selected_ensembles(store, actuals_path=path, run_dates=[day], only_missing=True).empty
    refreshed = build_selected_ensembles(store, actuals_path=path, run_dates=[day],
                                          only_missing=True, refresh_dates={day})
    assert len(refreshed) == 24
    assert len(store.read_forecasts().query("model == 'ensemble_selected'")) == 24


def _add_model(frame, model, offset):
    extra = frame[frame.model.eq('timesfm3_online')].copy()
    extra['model'] = model
    extra['forecast_mw'] += offset
    return pd.concat([frame, extra], ignore_index=True)


def _actual_file(tmp_path, day, actual):
    path = tmp_path / 'actuals.parquet'
    pd.DataFrame({'timestamp_utc': pd.date_range(day, periods=24, freq='h', tz='UTC'),
                  'zone': 'DE', 'target': 'wind_offshore', 'actual_mw': actual}).to_parquet(path)
    return path


def test_new_foundation_models_change_mean_and_invalidate_saved_membership(tmp_path):
    store = ForecastStore(data_dir=tmp_path / 'data')
    day = '2026-07-24'
    actual = np.linspace(100., 200., 24)
    frame = _component_rows(day, 'DE', 'wind_offshore', actual)
    path = _actual_file(tmp_path, day, actual)
    store.append_forecasts(frame, day)
    original = build_selected_ensembles(store, actuals_path=path, run_dates=[day])
    expanded = _add_model(_add_model(frame, 't0_gamma_online', 20), 'timestral_online', 40)
    store.append_forecasts(expanded, day)
    result = build_selected_ensembles(store, actuals_path=path, run_dates=[day], only_missing=True)
    assert len(result) == 24
    np.testing.assert_allclose(result.forecast_mw, actual + (10-5+2+22+42)/5)
    assert not np.allclose(result.forecast_mw, original.forecast_mw)
    assert result.ensemble_components.str.contains('t0_gamma_online').all()
    assert result.ensemble_components.str.contains('timestral_online').all()
    assert build_selected_ensembles(store, actuals_path=path, run_dates=[day], only_missing=True).empty


def test_future_registry_model_is_included_without_code_change(tmp_path, monkeypatch):
    from rt_forecast_dashboard.pipeline import build_selected_ensembles as module
    registry = module.model_registry()
    registry['future_model'] = dict(enabled=True, label='Future model', targets=['wind_offshore'])
    registry['disabled_model'] = dict(enabled=False, label='Disabled', targets=['wind_offshore'])
    registry['load_only_model'] = dict(enabled=True, label='Load only', targets=['load'])
    monkeypatch.setattr(module, 'model_registry', lambda: registry)
    day = '2026-07-24'
    actual = np.linspace(100., 200., 24)
    frame = _add_model(_component_rows(day, 'DE', 'wind_offshore', actual), 'future_model', 30)
    # Excluded and disabled models must not sneak into membership.
    frame = _add_model(frame, 'tabpfn_online', 1e6)
    frame = _add_model(frame, 'persistence', 1e6)
    frame = _add_model(frame, 'disabled_model', 1e6)
    frame = _add_model(frame, 'load_only_model', 1e6)
    store = ForecastStore(data_dir=tmp_path / 'data')
    store.append_forecasts(frame, day)
    result = module.build_selected_ensembles(store, actuals_path=_actual_file(tmp_path, day, actual), run_dates=[day])
    np.testing.assert_allclose(result.forecast_mw, actual + (10-5+2+32)/4)
    assert result.ensemble_components.str.contains('future_model').all()
    assert not result.ensemble_components.str.contains('tabpfn|persistence|tso_reference|disabled_model|load_only_model').any()


def test_incomplete_new_component_is_omitted(tmp_path):
    day = '2026-07-24'
    actual = np.linspace(100., 200., 24)
    frame = _add_model(_component_rows(day, 'DE', 'wind_offshore', actual), 't0_gamma_online', 20)
    frame.loc[frame.model.eq('t0_gamma_online') & (frame.index == frame.index.max()), 'forecast_mw'] = np.inf
    store = ForecastStore(data_dir=tmp_path / 'data')
    store.append_forecasts(frame, day)
    result = build_selected_ensembles(store, actuals_path=_actual_file(tmp_path, day, actual), run_dates=[day])
    np.testing.assert_allclose(result.forecast_mw, actual + (10-5+2)/3)
    assert not result.ensemble_components.str.contains('t0_gamma_online').any()


def test_expanded_learned_weights_use_prior_history_and_all_five_models(tmp_path):
    store = ForecastStore(data_dir=tmp_path / 'data')
    actual = np.linspace(100., 200., 24)
    outcomes = []
    days = pd.date_range('2026-07-24', periods=22)
    for date in days:
        day = str(date.date())
        frame = _component_rows(day, 'BE', 'load', actual)
        frame = _add_model(_add_model(frame, 't0_gamma_online', -2), 'timestral_online', -3)
        frame['actual_mw'] = np.tile(actual, len(frame)//24)
        store.append_forecasts(frame, day)
        outcomes.append(pd.DataFrame({'timestamp_utc': pd.date_range(day, periods=24, freq='h', tz='UTC'),
                                     'zone': 'BE', 'target': 'load', 'actual_mw': actual}))
    path = tmp_path / 'actuals.parquet'
    pd.concat(outcomes).to_parquet(path)
    early = build_selected_ensembles(store, actuals_path=path, run_dates=[str(days[0].date())])
    assert early.covariate_case.str.startswith('equal_weight_warmup:').all()
    latest = str(days[-1].date())
    result = build_selected_ensembles(store, actuals_path=path, run_dates=[latest])
    assert result.covariate_case.str.startswith('rolling_prior_actuals:').all()
    for block in result.ensemble_weights.iloc[0].split('|'):
        pairs = block.split(':', 1)[1].split(',')
        weights = [float(pair.rsplit('=', 1)[1]) for pair in pairs]
        assert len(weights) == 5
        assert abs(sum(weights)-1) < .0003
        assert all(0 <= weight <= .7001 for weight in weights)
        assert any(pair.startswith('T0 Gamma=') for pair in pairs)
        assert any(pair.startswith('Timestral=') for pair in pairs)
    # A shorter collector snapshot must not erase older stored outcomes or training.
    outcomes[-1].to_parquet(path)
    shortened = build_selected_ensembles(store, actuals_path=path, run_dates=[latest])
    assert shortened.covariate_case.str.startswith('rolling_prior_actuals:').all()
    np.testing.assert_allclose(result.forecast_mw, shortened.forecast_mw)
    np.testing.assert_allclose(shortened.actual_mw.astype(float), actual)
    changed = pd.concat(outcomes)
    changed.loc[changed.timestamp_utc.ge(pd.Timestamp(latest, tz='UTC')), 'actual_mw'] = 1e9
    changed.to_parquet(path)
    rerun = build_selected_ensembles(store, actuals_path=path, run_dates=[latest])
    np.testing.assert_allclose(result.forecast_mw, rerun.forecast_mw)
