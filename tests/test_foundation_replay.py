import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("replay", Path(__file__).resolve().parents[1] / "scripts/replay_foundation_online.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def inputs():
    times = pd.date_range("2026-04-23T16:00Z", periods=2238, freq="h")
    actual = pd.Series(np.arange(2238, dtype=float), index=times)
    tso = actual + 5
    weather = pd.DataFrame({"temperature_2m_p1": np.arange(2238, dtype=float)}, index=times)
    template = pd.DataFrame({"timestamp": times[-30:], "tso_forecast_mw": tso.iloc[-30:].to_numpy()})
    return template, actual, tso, weather


def test_replay_context_excludes_delivery_actuals():
    template, actual, tso, weather = inputs()
    actual.iloc[-30:] = 1e9
    x, cov, saved = replay.prepare(template, actual, tso, weather, ["temperature_2m_p1", "tso_forecast_mw"])
    np.testing.assert_array_equal(x, np.arange(2208))
    assert cov.shape == (2, 2238)
    assert len(saved) == 30
    assert (saved.actual_mw == 1e9).all()
    assert saved.context_end.iloc[0] < saved.timestamp.min()


def test_replay_rejects_missing_weather_boundary_and_incomplete_horizon():
    template, actual, tso, weather = inputs()
    weather.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing weather"):
        replay.prepare(template, actual, tso, weather, ["temperature_2m_p1"])
    with pytest.raises(ValueError, match="Incomplete"):
        replay.prepare(template.iloc[:-1], actual, tso, weather, ["temperature_2m_p1"])
