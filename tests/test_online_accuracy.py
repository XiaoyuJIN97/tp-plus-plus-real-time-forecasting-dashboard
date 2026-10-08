from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.ui.analytics import (
    online_forecast_accuracy,
    online_rmae_leaderboard,
    online_win_rate_by_zone,
)
from rt_forecast_dashboard.ui.charts import accuracy_summary_chart, deterministic_forecast_chart


def test_selected_ensemble_accuracy_combines_daily_weight_metadata() -> None:
    forecasts = pd.DataFrame(
        {
            "run_date": ["2026-10-01", "2026-10-02"],
            "zone": ["BE", "BE"],
            "model": ["ensemble_selected", "ensemble_selected"],
            "model_label": ["Ensembled model", "Ensembled model"],
            "covariate_case": ["weights for day 1", "weights for day 2"],
            "forecast_mw": [100.0, 110.0],
            "actual_mw": [102.0, 108.0],
        }
    )

    result = online_forecast_accuracy(forecasts)

    assert len(result) == 1
    assert result.loc[0, "case"] == "Selected configuration"
    assert result.loc[0, "n"] == 2
    assert result.loc[0, "MAE"] == 2.0


def test_online_accuracy_uses_dates_shared_by_every_model() -> None:
    rows = []
    for run_date, model, label, error in [
        ("2026-10-01", "ensemble_selected", "Ensembled model", 1.0),
        ("2026-10-01", "timesfm3_online", "TimesFM3", 2.0),
        ("2026-10-02", "timesfm3_online", "TimesFM3", 100.0),
    ]:
        for horizon in range(24):
            rows.append(
                {
                    "run_date": run_date,
                    "zone": "BE",
                    "target": "load",
                    "model": model,
                    "model_label": label,
                    "covariate_case": "test",
                    "horizon": horizon,
                    "forecast_mw": 100.0 + error,
                    "actual_mw": 100.0,
                }
            )

    result = online_forecast_accuracy(pd.DataFrame(rows)).set_index("display_family")

    assert result.loc["Ensembled model", "n"] == 24
    assert result.loc["TimesFM3", "n"] == 24
    assert result.loc["TimesFM3", "MAE"] == 2.0


def test_rmae_leaderboard_uses_paired_complete_runs() -> None:
    rows = []
    for model, label, error in [
        ("timesfm3_online", "TimesFM3 with covariates", 5.0),
        ("tso_reference", "TSO forecast", 10.0),
    ]:
        for horizon in range(24):
            rows.append(
                {
                    "run_date": "2026-10-01",
                    "zone": "BE",
                    "target": "load",
                    "model": model,
                    "model_label": label,
                    "horizon": horizon,
                    "forecast_mw": 100.0 + error,
                    "actual_mw": 100.0,
                }
            )
    detail, summary = online_rmae_leaderboard(pd.DataFrame(rows))

    timesfm = detail[detail["display_family"].eq("TimesFM3")].iloc[0]
    tso = detail[detail["display_family"].eq("TSO forecast")].iloc[0]
    assert timesfm["rMAE"] == 0.5
    assert tso["rMAE"] == 1.0
    assert summary.iloc[0]["display_family"] == "TimesFM3"
    assert summary.iloc[0]["rank"] == 1


def test_accuracy_chart_uses_model_family_axis_titles() -> None:
    frame = pd.DataFrame(
        {
            "country": ["BE"],
            "display_family": ["TimesFM3"],
            "MAE": [10.0],
        }
    )

    figure = accuracy_summary_chart(frame, "MAE")

    assert figure.layout.xaxis.title.text == "Model Family"
    assert figure.layout.legend.title.text == "Model Family"


def test_forecast_chart_uses_market_timezone_label_and_reserves_footer_space() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-26T00:00:00Z", "2026-09-26T01:00:00Z"]),
            "run_date": ["2026-09-25", "2026-09-25"],
            "zone": ["BE", "BE"],
            "model_label": ["TimesFM3", "TimesFM3"],
            "forecast_mw": [100.0, 110.0],
        }
    )

    figure = deterministic_forecast_chart(frame, "Load forecast")

    assert figure.layout.xaxis.title.text == "Delivery time (CET/CEST)"
    assert figure.layout.height >= 500
    assert figure.layout.margin.b >= 150
    assert figure.layout.xaxis.automargin is True


def test_win_rate_is_reported_per_zone_with_half_credit_for_ties() -> None:
    rows = []
    errors = {
        "BE": [(5.0, 10.0), (10.0, 10.0), (15.0, 10.0)],
        "DE": [(5.0, 10.0), (5.0, 10.0), (5.0, 10.0)],
    }
    for zone, daily_errors in errors.items():
        for day, (model_error, tso_error) in enumerate(daily_errors, start=1):
            for model, label, error in [
                ("timesfm3_online", "TimesFM3 with covariates", model_error),
                ("tso_reference", "TSO forecast", tso_error),
            ]:
                for horizon in range(24):
                    rows.append(
                        {
                            "run_date": f"2026-10-0{day}",
                            "zone": zone,
                            "target": "load",
                            "model": model,
                            "model_label": label,
                            "horizon": horizon,
                            "forecast_mw": 100.0 + error,
                            "actual_mw": 100.0,
                        }
                    )

    result = online_win_rate_by_zone(pd.DataFrame(rows))
    timesfm = result[result["display_family"].eq("TimesFM3")].set_index("zone")

    assert timesfm.loc["BE", "compared_runs"] == 3
    assert timesfm.loc["BE", "win_rate"] == 50.0
    assert timesfm.loc["DE", "win_rate"] == 100.0
