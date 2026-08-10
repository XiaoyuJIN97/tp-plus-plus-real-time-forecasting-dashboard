from __future__ import annotations

from rt_forecast_dashboard.config import model_registry
from rt_forecast_dashboard.models.adapters import make_adapter


def enabled_models_for_target(target: str) -> dict[str, dict]:
    registry = model_registry()
    return {
        key: config
        for key, config in registry.items()
        if config.get("enabled", False) and target in config.get("targets", [])
    }


def iter_model_adapters(target: str):
    for key, config in enabled_models_for_target(target).items():
        yield key, config, make_adapter(config)
