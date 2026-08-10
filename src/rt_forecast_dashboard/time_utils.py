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


def latest_complete_run_date(now: datetime | None = None) -> date:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    local_now = now.astimezone(BRUSSELS_TZ)
    run_day = local_now.date()
    if local_now.hour < DAILY_CUTOFF_HOUR:
        run_day = run_day - timedelta(days=1)
    return run_day
