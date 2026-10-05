from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from rt_forecast_dashboard.config import source_paths


REQUIRED_COLUMNS = [
    "timestamp_utc",
    "collection_time_utc",
    "zone",
    "target",
    "actual_mw",
]


def fetch_dashboard_actuals(timeout_seconds: float = 10) -> pd.DataFrame:
    """Read the compact static actuals artifact published by the collector."""
    url = source_paths().get("entsoe_dashboard_actuals_url", "")
    if not url:
        raise RuntimeError("ENTSO-E dashboard actuals URL is not configured.")
    request = Request(
        url,
        headers={"User-Agent": "tp-plus-plus-real-time-forecasting-dashboard"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not read the static ENTSO-E actuals artifact: {exc}") from exc

    frame = pd.read_parquet(BytesIO(payload))
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(
            "Static ENTSO-E actuals artifact is missing columns: " + ", ".join(sorted(missing))
        )
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame.pop("timestamp_utc"), utc=True, errors="coerce")
    frame["collection_time_utc"] = pd.to_datetime(
        frame["collection_time_utc"], utc=True, errors="coerce"
    )
    frame["actual_mw"] = pd.to_numeric(frame["actual_mw"], errors="coerce")
    return frame.dropna(subset=["timestamp", "actual_mw"]).sort_values(
        ["zone", "target", "timestamp"]
    )
