from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from rt_forecast_dashboard.config import load_settings
from rt_forecast_dashboard.data.demo import demo_context_frame, demo_future_frame
from rt_forecast_dashboard.data.weather_archive import OpenMeteoRealtimeArchive
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp

OPEN_METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_100m",
    "wind_direction_100m",
]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


class OpenMeteoClient:
    def __init__(self, demo_mode: bool | None = None) -> None:
        settings = load_settings()
        self.demo_mode = settings.demo_mode if demo_mode is None else demo_mode
        self.archive = OpenMeteoRealtimeArchive()

    def fetch_future_weather(self, *, run_date: date, zone: str, points: list[dict], target: str, horizon_hours: int) -> pd.DataFrame:
        if self.demo_mode:
            return demo_future_frame(run_date, zone, target, horizon_hours).drop(columns=["actual_mw", "tso_forecast_mw"], errors="ignore")
        start = brussels_cutoff_timestamp(run_date)
        archive = self.archive.fetch_weather(zone=zone, target=target, start=start, hours=horizon_hours)
        if len(archive) >= horizon_hours:
            return archive.head(horizon_hours)
        return self._fetch_openmeteo(points=points, start=start, hours=horizon_hours, historical=False)

    def fetch_context_weather(self, *, run_date: date, zone: str, points: list[dict], target: str, context_hours: int) -> pd.DataFrame:
        if self.demo_mode:
            return demo_context_frame(run_date, zone, target, context_hours).drop(columns=["actual_mw", "tso_forecast_mw"])
        end = brussels_cutoff_timestamp(run_date)
        start = end - pd.Timedelta(hours=context_hours)
        archive = self.archive.fetch_weather(zone=zone, target=target, start=start, hours=context_hours)
        if len(archive) >= context_hours:
            return archive.head(context_hours)
        return self._fetch_openmeteo(points=points, start=start, hours=context_hours, historical=True)

    def _fetch_openmeteo(self, *, points: list[dict], start: pd.Timestamp, hours: int, historical: bool) -> pd.DataFrame:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=retry_session)
        frames = []
        start_date = start.date()
        end_date = (start + pd.Timedelta(hours=hours + 24)).date()
        for point in points:
            response = client.weather_api(
                HISTORICAL_FORECAST_URL if historical else FORECAST_URL,
                params={
                    "latitude": point["gfs_lat"],
                    "longitude": point["gfs_lon"],
                    "hourly": OPEN_METEO_VARIABLES,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "timezone": "UTC",
                    "wind_speed_unit": "ms",
                },
                timeout=20,
            )[0]
            hourly = response.Hourly()
            timestamps = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
            point_no = int(point["point"])
            frame = pd.DataFrame({"timestamp": timestamps})
            for idx, variable in enumerate(OPEN_METEO_VARIABLES):
                values = hourly.Variables(idx).ValuesAsNumpy()
                if variable == "wind_speed_100m":
                    frame[f"wind_speed_100m_ms_p{point_no}"] = values
                elif variable == "wind_direction_100m":
                    frame[f"wind_dir_sin_p{point_no}"] = np.sin(np.deg2rad(values))
                    frame[f"wind_dir_cos_p{point_no}"] = np.cos(np.deg2rad(values))
                else:
                    frame[f"{variable}_p{point_no}"] = values
            frames.append(frame)
        weather = frames[0]
        for frame in frames[1:]:
            weather = weather.merge(frame, on="timestamp", how="outer")
        weather = weather.sort_values("timestamp")
        mask = (weather["timestamp"] >= start) & (weather["timestamp"] < start + pd.Timedelta(hours=hours))
        weather = weather.loc[mask].copy()
        temp_cols = [c for c in weather.columns if c.startswith("temperature_2m_p")]
        if temp_cols:
            weather["deg_proxy"] = (weather[temp_cols].mean(axis=1) - 18.0).abs()
        return weather.head(hours)
