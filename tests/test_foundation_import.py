import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rt_forecast_dashboard.storage import ForecastStore

spec = importlib.util.spec_from_file_location("replay_import", Path(__file__).resolve().parents[1] / "scripts/import_foundation_replay.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_import_keeps_forecasts_and_refreshes_shared_actuals(tmp_path, monkeypatch):
    store = ForecastStore(tmp_path / "production")
    day = "2026-07-24"
    old = pd.DataFrame({"run_date": day, "run_at": "2026-07-24T16:00Z", "zone": "BE", "target": "load", "model": "timesfm3_online", "timestamp": pd.date_range("2026-07-24T16:00Z", periods=24, freq="h"), "forecast_mw": np.arange(24) + 0.123456789, "actual_mw": 1.0})
    store.append_forecasts(old, day)
    replay = tmp_path / "replay"
    (replay / "forecasts").mkdir(parents=True)
    new = old.copy()
    new["model"] = "t0_gamma_online"
    new["forecast_mw"] = 42.0
    new["actual_mw"] = 2.0
    new["source"] = "historical_mock_replay"
    new.to_csv(replay / "forecasts" / f"forecasts_{day}.csv", index=False)
    monkeypatch.setattr("sys.argv", ["import", "--replay", str(replay), "--data-dir", str(store.data_dir)])
    module.main()
    after = store.read_forecasts()
    original = after[after.model.eq("timesfm3_online")]
    np.testing.assert_allclose(original.forecast_mw, old.forecast_mw)
    assert original.actual_mw.eq(2).all()
    assert len(after) == 48
    assert json.loads((replay / "import_validation.json").read_text())["predictions_preserved"]
