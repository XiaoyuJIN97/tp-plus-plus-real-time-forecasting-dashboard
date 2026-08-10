from __future__ import annotations

from rt_forecast_dashboard.config import load_optimal_engineering_covariates

LOAD_GROUPS = {
    "Temp": ["temperature_2m_p1", "temperature_2m_p2", "temperature_2m_p3", "temperature_2m_p4"],
    "Hum": ["relative_humidity_2m_p1", "relative_humidity_2m_p2", "relative_humidity_2m_p3", "relative_humidity_2m_p4"],
    "Solar": ["shortwave_radiation_p1", "shortwave_radiation_p2", "shortwave_radiation_p3", "shortwave_radiation_p4"],
    "deg_proxy": ["deg_proxy"],
}

SOLAR_COVARIATES = [
    "shortwave_radiation_p1",
    "shortwave_radiation_p2",
    "shortwave_radiation_p3",
    "shortwave_radiation_p4",
    "temperature_2m_p1",
    "temperature_2m_p2",
    "temperature_2m_p3",
    "temperature_2m_p4",
]

WIND_COVARIATES = [
    "wind_speed_100m_ms_p1",
    "wind_speed_100m_ms_p2",
    "wind_speed_100m_ms_p3",
    "wind_speed_100m_ms_p4",
    "wind_dir_sin_p1",
    "wind_dir_sin_p2",
    "wind_dir_sin_p3",
    "wind_dir_sin_p4",
    "wind_dir_cos_p1",
    "wind_dir_cos_p2",
    "wind_dir_cos_p3",
    "wind_dir_cos_p4",
]


def covariates_for(target: str, country: str, model_key: str) -> tuple[str, list[str]]:
    if model_key == "tso_reference":
        return "tso_forecast", ["tso_forecast_mw"]
    if model_key == "persistence":
        return "weekly_persistence" if target == "load" else "daily_persistence", []

    base_model = model_key.split("_", 1)[0].capitalize()
    if model_key.startswith("ridge"):
        base_model = "Ridge"
    elif model_key.startswith("chronos"):
        base_model = "Chronos2"
    elif model_key.startswith("tabpfn"):
        base_model = "TabPFN"
    elif model_key.startswith("xgboost"):
        base_model = "XGBoost"

    if target == "load":
        selected = load_optimal_engineering_covariates().get(base_model, {}).get(country, "Temp")
        return f"{selected}+tso_forecast", LOAD_GROUPS[selected] + ["tso_forecast_mw"]
    if target == "solar":
        return "weather_4p+tso_forecast", SOLAR_COVARIATES + ["tso_forecast_mw"]
    if target in {"wind_onshore", "wind_offshore"}:
        return "weather_4p+tso_forecast", WIND_COVARIATES + ["tso_forecast_mw"]
    return "tso_forecast", ["tso_forecast_mw"]
