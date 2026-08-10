from __future__ import annotations

from pathlib import Path

import pandas as pd

from rt_forecast_dashboard.config import source_paths


def selected_weather_points(point_type: str, country: str) -> list[dict]:
    paths = source_paths()
    selected_path = Path(paths["selected_weather_points"])
    if point_type == "load" and Path(paths.get("load_selected_weather_points", "")).exists():
        selected_path = Path(paths["load_selected_weather_points"])
    points = pd.read_csv(selected_path)
    subset = points[(points["country"] == country) & (points["type"] == point_type)].copy()
    subset = subset.sort_values("point")
    if subset.empty:
        raise ValueError(f"No selected weather points for {country} {point_type}")
    return subset.to_dict("records")
