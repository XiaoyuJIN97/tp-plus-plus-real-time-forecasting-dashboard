from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.pipeline.check_forecast_complete import expected_groups, is_forecast_complete


def _forecast_rows(groups: set[tuple[str, str, str]], hours: int = 24) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range("2026-08-30T16:00:00Z", periods=hours, freq="h")
    for zone, target, model in groups:
        for timestamp in timestamps:
            rows.append(
                {
                    "run_date": "2026-08-30",
                    "zone": zone,
                    "target": target,
                    "model": model,
                    "timestamp": timestamp.isoformat(),
                    "forecast_mw": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_forecast_completeness_accepts_full_core_file(tmp_path) -> None:
    path = tmp_path / "forecasts" / "forecasts_2026-08-30.csv"
    path.parent.mkdir()
    model_keys = {"tso_reference", "persistence", "ridge_3mo_context", "chronos2_online", "xgboost_online"}
    _forecast_rows(expected_groups(model_keys)).to_csv(path, index=False)

    complete, reason = is_forecast_complete(path, model_keys)

    assert complete
    assert "complete" in reason


def test_forecast_completeness_rejects_partial_file(tmp_path) -> None:
    path = tmp_path / "forecasts" / "forecasts_2026-08-30.csv"
    path.parent.mkdir()
    partial = {
        ("BE", "load", "tso_reference"),
        ("BE", "load", "persistence"),
        ("BE", "load", "ridge_3mo_context"),
        ("BE", "load", "chronos2_online"),
    }
    _forecast_rows(partial).to_csv(path, index=False)

    complete, reason = is_forecast_complete(path, {"tso_reference", "persistence", "ridge_3mo_context", "chronos2_online", "xgboost_online"})

    assert not complete
    assert "incomplete groups" in reason


def test_forecast_completeness_rejects_stale_context(tmp_path) -> None:
    path = tmp_path / "forecasts" / "forecasts_2026-08-30.csv"
    path.parent.mkdir()
    model_keys = {"tso_reference"}
    frame = _forecast_rows(expected_groups(model_keys))
    frame["context_end"] = "2026-08-30T14:00:00Z"
    frame.to_csv(path, index=False)

    complete, reason = is_forecast_complete(path, model_keys)

    assert not complete
    assert "stale context" in reason


def test_recovery_runs_newest_first_and_only_retries_missing_groups(tmp_path,monkeypatch):
    from rt_forecast_dashboard.pipeline import backfill
    from rt_forecast_dashboard.storage import ForecastStore
    calls=[]
    monkeypatch.setattr(backfill,'ForecastStore',lambda:ForecastStore(data_dir=tmp_path))
    monkeypatch.setattr(backfill,'_completion_status',lambda *args:(False,'missing test group'))
    def run(day,model_keys=None,**kwargs):
        calls.append((day.isoformat(),kwargs['only_missing']))
        return pd.DataFrame({'forecast_mw':[1.]})
    monkeypatch.setattr(backfill,'run_daily_forecast',run)
    backfill.run_backfill('2026-10-26','2026-10-28',{'tso_reference'})
    assert calls==[('2026-10-28',True),('2026-10-27',True),('2026-10-26',True)]
    calls.clear()
    backfill.run_backfill('2026-10-28','2026-10-28',{'tso_reference'},force=True)
    assert calls==[('2026-10-28',False)]
