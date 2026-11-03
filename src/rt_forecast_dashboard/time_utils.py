from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

BRUSSELS_TZ = ZoneInfo("Europe/Brussels")
DAILY_CUTOFF_HOUR = 18


def brussels_cutoff_datetime(run_date: date) -> datetime:
    return datetime.combine(run_date, time(hour=DAILY_CUTOFF_HOUR), tzinfo=BRUSSELS_TZ)


def brussels_cutoff_timestamp(run_date: date) -> pd.Timestamp:
    return pd.Timestamp(brussels_cutoff_datetime(run_date)).tz_convert("UTC")


def forecast_end_timestamp(run_date: date | str) -> pd.Timestamp:
    """Exclusive delivery end: midnight after the following Brussels day."""
    day = pd.Timestamp(run_date).date()
    return pd.Timestamp(datetime.combine(day + timedelta(days=2), time(), tzinfo=BRUSSELS_TZ)).tz_convert("UTC")


def forecast_horizon_hours(run_date: date | str, nominal_hours: int = 30) -> int:
    """18:00 origin through the following day, normally 30 hours (29/31 at DST).

    Explicit experimental lengths retain their configured size. Production
    configurations use 30; compute elapsed time between timezone-aware endpoints.
    """
    if nominal_hours != 30:
        return nominal_hours
    day = pd.Timestamp(run_date).date()
    return int((forecast_end_timestamp(day) - brussels_cutoff_timestamp(day)) / pd.Timedelta(hours=1))


def forecast_delivery_grid(run_date: date | str, nominal_hours: int = 30) -> pd.DatetimeIndex:
    """Exact UTC delivery instants, including the final hour ending at midnight."""
    day = pd.Timestamp(run_date).date()
    return pd.date_range(brussels_cutoff_timestamp(day),
                         periods=forecast_horizon_hours(day, nominal_hours), freq="h")


def is_complete_forecast_window(timestamps, run_date: date | str, nominal_hours: int = 30) -> bool:
    """Reject shifted, duplicated, missing or legacy delivery timestamps."""
    observed = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce")).sort_values()
    return observed.equals(forecast_delivery_grid(run_date, nominal_hours))


def latest_complete_run_date(now: datetime | None = None) -> date:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    local_now = now.astimezone(BRUSSELS_TZ)
    run_day = local_now.date()
    if local_now.hour < DAILY_CUTOFF_HOUR:
        run_day = run_day - timedelta(days=1)
    return run_day
