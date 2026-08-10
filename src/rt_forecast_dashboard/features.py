from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(tso: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    frame = pd.merge(tso, weather, on="timestamp", how="outer").sort_values("timestamp")
    frame["tso_forecast_mw"] = frame["tso_forecast_mw"].interpolate(limit_direction="both")
    for column in frame.columns:
        if column != "timestamp":
            frame[column] = frame[column].interpolate(limit_direction="both")
    dt = pd.to_datetime(frame["timestamp"])
    frame["hour"] = dt.dt.hour
    frame["dayofweek"] = dt.dt.dayofweek
    frame["month"] = dt.dt.month
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    frame["daylight_proxy"] = np.maximum(0, np.sin((frame["hour"] - 6) / 13 * np.pi))
    return frame
