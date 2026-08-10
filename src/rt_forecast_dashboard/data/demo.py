from __future__ import annotations

import math
import hashlib
from datetime import date

import numpy as np
import pandas as pd

from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp


def cutoff_timestamp(run_date: date) -> pd.Timestamp:
    return brussels_cutoff_timestamp(run_date)


def _base_level(zone: str, target: str) -> float:
    zone_factor = {"BE": 1.0, "FR": 2.7, "DE": 3.4}.get(zone, 1.0)
    return {"load": 6500, "solar": 900, "wind_onshore": 1500, "wind_offshore": 950}[target] * zone_factor


def _target_values(timestamps: pd.DatetimeIndex, zone: str, target: str, seed_key: str) -> np.ndarray:
    base = _base_level(zone, target)
    zone_factor = {"BE": 0.2, "FR": 0.7, "DE": 1.1}.get(zone, 0.2)
    idx = np.arange(len(timestamps))
    hours = np.array([ts.hour for ts in timestamps])
    if target == "solar":
        daylight = np.maximum(0.0, np.sin((hours - 6) / 13 * math.pi))
        values = base * daylight * (0.8 + 0.18 * np.sin(idx / 37 + zone_factor))
    elif target == "wind_onshore":
        values = base * (0.85 + 0.28 * np.sin(idx / 15 + zone_factor) + 0.12 * np.sin(idx / 71))
    elif target == "wind_offshore":
        values = base * (0.9 + 0.24 * np.sin(idx / 18 + zone_factor) + 0.10 * np.cos(idx / 67))
    else:
        daily = np.sin((hours - 7) / 24 * 2 * math.pi)
        weekly = 0.06 * np.sin(idx / 168 * 2 * math.pi + zone_factor)
        values = base * (1.0 + 0.12 * daily + weekly + 0.05 * np.isin(hours, [8, 9, 18, 19]))
    noise = _timestamp_noise(timestamps, f"{zone}:{target}", base * 0.015)
    return np.maximum(values + noise, 0.0)


def _timestamp_noise(timestamps: pd.DatetimeIndex, series_key: str, scale: float) -> np.ndarray:
    values = []
    for ts in timestamps:
        payload = f"{series_key}:{pd.Timestamp(ts).isoformat()}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        unit = int.from_bytes(digest, "big") / 2**64
        values.append((unit - 0.5) * 2.0 * scale)
    return np.asarray(values, dtype=float)


def weather_frame(timestamps: pd.DatetimeIndex, zone: str, target: str) -> pd.DataFrame:
    idx = np.arange(len(timestamps))
    hours = np.array([ts.hour for ts in timestamps])
    zone_shift = {"BE": 0.0, "FR": 0.6, "DE": 1.2}.get(zone, 0.0)
    frame = pd.DataFrame({"timestamp": timestamps})
    for point in range(1, 5):
        shift = zone_shift + point * 0.35
        temp = 15 + 8 * np.sin(idx / 24 * 2 * np.pi + shift) + 2 * np.sin(idx / 168 * 2 * np.pi)
        hum = np.clip(65 - 18 * np.sin(idx / 24 * 2 * np.pi + shift) + 8 * np.cos(idx / 80), 25, 98)
        solar = np.maximum(0, 760 * np.sin((hours - 6) / 13 * math.pi)) * (0.75 + 0.2 * np.sin(idx / 19 + shift))
        wind_speed = np.maximum(0.5, 7 + 2.2 * np.sin(idx / 13 + shift) + 1.5 * np.sin(idx / 61))
        wind_dir = (180 + 70 * np.sin(idx / 31 + shift) + point * 15) % 360
        frame[f"temperature_2m_p{point}"] = temp
        frame[f"relative_humidity_2m_p{point}"] = hum
        frame[f"shortwave_radiation_p{point}"] = solar
        frame[f"wind_speed_100m_ms_p{point}"] = wind_speed
        frame[f"wind_dir_sin_p{point}"] = np.sin(np.deg2rad(wind_dir))
        frame[f"wind_dir_cos_p{point}"] = np.cos(np.deg2rad(wind_dir))
    frame["deg_proxy"] = (frame[[f"temperature_2m_p{i}" for i in range(1, 5)]].mean(axis=1) - 18.0).abs()
    return frame


def tso_forecast_values(actual_values: np.ndarray, zone: str, target: str, seed_key: str) -> np.ndarray:
    base = _base_level(zone, target)
    rng = np.random.default_rng(abs(hash((seed_key, "tso"))) % (2**32))
    bias = {"load": 0.006, "solar": -0.015, "wind_onshore": 0.018, "wind_offshore": -0.01}[target]
    smooth_error = np.sin(np.arange(len(actual_values)) / 11) * base * 0.02
    return np.maximum(actual_values * (1 + bias) + smooth_error + rng.normal(0, base * 0.02, len(actual_values)), 0.0)


def demo_future_frame(run_date: date, zone: str, target: str, horizon_hours: int) -> pd.DataFrame:
    start = cutoff_timestamp(run_date)
    timestamps = pd.date_range(start=start, periods=horizon_hours, freq="h")
    actual = _target_values(timestamps, zone, target, f"{run_date}:{zone}:{target}:future")
    frame = weather_frame(timestamps, zone, target)
    frame["actual_mw"] = actual
    frame["tso_forecast_mw"] = tso_forecast_values(actual, zone, target, f"{run_date}:{zone}:{target}:future")
    return frame


def demo_context_frame(run_date: date, zone: str, target: str, context_hours: int) -> pd.DataFrame:
    end = cutoff_timestamp(run_date)
    timestamps = pd.date_range(end=end - pd.Timedelta(hours=1), periods=context_hours, freq="h")
    actual = _target_values(timestamps, zone, target, f"{run_date}:{zone}:{target}:context")
    frame = weather_frame(timestamps, zone, target)
    frame["actual_mw"] = actual
    frame["tso_forecast_mw"] = tso_forecast_values(actual, zone, target, f"{run_date}:{zone}:{target}:context")
    return frame
