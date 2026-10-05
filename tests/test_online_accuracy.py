from __future__ import annotations

import pandas as pd

from rt_forecast_dashboard.ui.analytics import online_forecast_accuracy


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
