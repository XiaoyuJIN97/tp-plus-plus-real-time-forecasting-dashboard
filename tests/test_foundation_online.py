import numpy as np
import pandas as pd
import pytest

from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.config import zones
from rt_forecast_dashboard.models.adapters import make_adapter
from rt_forecast_dashboard.models.foundation_online import FoundationOnlineAdapter
from rt_forecast_dashboard.pipeline.check_forecast_complete import expected_groups
from rt_forecast_dashboard.ui.analytics import online_enabled_model_families


@pytest.mark.parametrize("key", ["t0_gamma_online", "timestral_online"])
def test_same_covariate_policy_and_online_visibility(key):
    for zone, config in zones().items():
        for target in config.get("targets", ("load", "solar", "wind_onshore", "wind_offshore")):
            assert covariates_for(target, zone, key) == covariates_for(target, zone, "timesfm3_online")
            assert (zone, target, key) in expected_groups({key})
    family = "T0 Gamma" if key.startswith("t0") else "Timestral"
    assert family in online_enabled_model_families("load")
    assert isinstance(make_adapter({"adapter": key}), FoundationOnlineAdapter)


def frames():
    ts = pd.date_range("2026-07-01", periods=192, freq="h", tz="UTC")
    past = pd.DataFrame({"timestamp": ts[:168], "actual_mw": np.arange(168), "tso_forecast_mw": np.arange(168)})
    future = pd.DataFrame({"timestamp": ts[168:], "tso_forecast_mw": np.arange(168, 192)})
    return past, future


def test_context_and_future_are_separate_and_output_order_is_preserved(monkeypatch):
    past, future = frames()
    adapter = FoundationOnlineAdapter("t0_gamma")
    def predict(contexts, covariates, horizon):
        np.testing.assert_array_equal(contexts[0], np.arange(168))
        np.testing.assert_array_equal(covariates[0], np.arange(192)[None, :])
        return np.arange(horizon)[None, :]
    monkeypatch.setattr(adapter, "predict_batch", predict)
    result = adapter.predict(future.iloc[::-1], "load", context=past, covariates=["tso_forecast_mw"])
    np.testing.assert_array_equal(result, np.arange(24)[::-1])


def test_rejects_leaking_actuals_and_missing_covariates():
    past, future = frames()
    adapter = FoundationOnlineAdapter("timestral")
    with pytest.raises(ValueError, match="missing"):
        adapter.predict(future, "load", past, ["absent"])
    past.loc[167, "timestamp"] = future.timestamp.iloc[0]
    with pytest.raises(ValueError, match="overlaps"):
        adapter.predict(future, "load", past)
