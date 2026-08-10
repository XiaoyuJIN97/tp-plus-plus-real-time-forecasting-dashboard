from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    entsoe_api_key: str | None
    demo_mode: bool


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    data_dir = Path(os.getenv("FORECAST_DATA_DIR", PROJECT_ROOT / "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    demo_mode = os.getenv("FORECAST_DEMO_MODE", "false").lower() in {"1", "true", "yes"}
    return Settings(
        data_dir=data_dir,
        entsoe_api_key=os.getenv("ENTSOE_API_KEY") or None,
        demo_mode=demo_mode,
    )


def load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def zones() -> dict[str, dict[str, Any]]:
    return load_yaml("zones.yml")["zones"]


def features() -> dict[str, dict[str, Any]]:
    return load_yaml("features.yml")["targets"]


def forecast_runtime() -> dict[str, Any]:
    config = load_yaml("features.yml")
    return config["cutoff"]


def load_optimal_engineering_covariates() -> dict[str, dict[str, str]]:
    return load_yaml("features.yml")["load_optimal_engineering_covariates"]


def source_paths() -> dict[str, str]:
    paths = load_yaml("features.yml")["source_paths"]
    env_overrides = {
        "selected_weather_points": "SELECTED_WEATHER_POINTS",
        "load_selected_weather_points": "LOAD_SELECTED_WEATHER_POINTS",
        "weather_forecasts_4p": "WEATHER_FORECASTS_4P",
        "open_meteo_realtime_data": "OPEN_METEO_REALTIME_DATA",
        "open_meteo_realtime_git_ref": "OPEN_METEO_REALTIME_GIT_REF",
        "entsoe_realtime_data": "ENTSOE_REALTIME_DATA",
        "entsoe_realtime_git_ref": "ENTSOE_REALTIME_GIT_REF",
        "entsoe_realtime_raw_base": "ENTSOE_REALTIME_RAW_BASE",
    }
    for key, env_name in env_overrides.items():
        value = os.getenv(env_name)
        if value:
            paths[key] = value
    return paths


def model_registry() -> dict[str, dict[str, Any]]:
    return load_yaml("model_registry.yml")["models"]
