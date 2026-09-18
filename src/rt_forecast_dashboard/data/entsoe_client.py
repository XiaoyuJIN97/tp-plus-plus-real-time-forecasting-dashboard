from __future__ import annotations

from datetime import date

import pandas as pd

from rt_forecast_dashboard.config import load_settings
from rt_forecast_dashboard.data.demo import demo_context_frame, demo_future_frame
from rt_forecast_dashboard.data.entsoe_realtime_archive import EntsoeRealtimeArchive
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp


_STORED_FORECAST_CACHE: pd.DataFrame | None = None


def _entsoe_response_to_hourly_frame(response: pd.Series | pd.DataFrame, value_col: str) -> pd.DataFrame:
    if isinstance(response, pd.DataFrame):
        numeric = response.select_dtypes(include="number")
        values = numeric.sum(axis=1) if len(numeric.columns) > 1 else numeric.iloc[:, 0]
    else:
        values = response
    frame = values.rename(value_col).reset_index().rename(columns={"index": "timestamp"})
    timestamp_col = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={timestamp_col: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    hourly = (
        frame.dropna(subset=[value_col])
        .sort_values("timestamp")
        .set_index("timestamp")[value_col]
        .resample("h")
        .mean()
        .dropna()
        .reset_index()
    )
    return hourly[["timestamp", value_col]]


class EntsoeForecastClient:
    def __init__(self, api_key: str | None = None, demo_mode: bool | None = None) -> None:
        settings = load_settings()
        self.api_key = api_key if api_key is not None else settings.entsoe_api_key
        self.demo_mode = settings.demo_mode if demo_mode is None else demo_mode
        self.archive = EntsoeRealtimeArchive()

    def fetch_tso_forecast(self, *, run_date: date, zone: str, zone_code: str, target: str, horizon_hours: int) -> pd.DataFrame:
        start = brussels_cutoff_timestamp(run_date)
        end = start + pd.Timedelta(hours=horizon_hours)
        archive = self.archive.fetch_forecast(zone=zone, target=target, start=start, end=end)
        if len(archive) >= horizon_hours:
            return archive.head(horizon_hours)
        stored = self._stored_tso_forecast(run_date=run_date, zone=zone, target=target, start=start, end=end, horizon_hours=horizon_hours)
        if len(stored) >= horizon_hours:
            return stored.head(horizon_hours)
        if self.demo_mode:
            return demo_future_frame(run_date, zone, target, horizon_hours)[["timestamp", "tso_forecast_mw"]]
        if not self.api_key:
            raise RuntimeError(f"No realtime-data forecast coverage for {zone} {target}, and ENTSOE_API_KEY is not configured.")

        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=self.api_key)
        if target == "load":
            series = client.query_load_forecast(zone_code, start=start, end=end)
        elif target == "solar":
            series = client.query_wind_and_solar_forecast(zone_code, start=start, end=end, psr_type="B16")
        elif target == "wind_onshore":
            series = client.query_wind_and_solar_forecast(zone_code, start=start, end=end, psr_type="B19")
        elif target == "wind_offshore":
            series = client.query_wind_and_solar_forecast(zone_code, start=start, end=end, psr_type="B18")
        else:
            raise ValueError(f"Unsupported target: {target}")
        frame = _entsoe_response_to_hourly_frame(series, "tso_forecast_mw").head(horizon_hours)
        if len(frame) < horizon_hours:
            raise RuntimeError(f"Incomplete ENTSO-E TSO forecast horizon for {zone} {target}: {len(frame)}/{horizon_hours} hourly rows.")
        return frame

    def fetch_actual_context(self, *, run_date: date, zone: str, zone_code: str, target: str, context_hours: int) -> pd.DataFrame:
        end = brussels_cutoff_timestamp(run_date)
        start = end - pd.Timedelta(hours=context_hours + 72)
        actual_archive = self.archive.fetch_actuals(zone=zone, target=target, start=start, end=end)
        forecast_archive = self.archive.fetch_forecast(zone=zone, target=target, start=start, end=end)
        if len(actual_archive) >= context_hours and len(forecast_archive) >= context_hours:
            frame = actual_archive.merge(forecast_archive, on="timestamp", how="inner").sort_values("timestamp")
            frame = frame[["timestamp", "actual_mw", "tso_forecast_mw"]].tail(context_hours)
            expected_last = end - pd.Timedelta(hours=1)
            if len(frame) >= context_hours and pd.to_datetime(frame["timestamp"], utc=True).max() >= expected_last:
                return frame
            if len(frame) >= context_hours:
                return frame
        stored_context = self._stored_actual_context(zone=zone, target=target, start=start, end=end)
        if not stored_context.empty:
            context_parts = []
            if not actual_archive.empty and not forecast_archive.empty:
                context_parts.append(actual_archive.merge(forecast_archive, on="timestamp", how="inner"))
            context_parts.append(stored_context)
            frame = pd.concat(context_parts, ignore_index=True)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            for column in ["actual_mw", "tso_forecast_mw"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["actual_mw", "tso_forecast_mw"])
            frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
            if len(frame) >= context_hours:
                return frame[["timestamp", "actual_mw", "tso_forecast_mw"]].tail(context_hours)
        if self.demo_mode:
            return demo_context_frame(run_date, zone, target, context_hours)[["timestamp", "actual_mw", "tso_forecast_mw"]]
        if not self.api_key:
            raise RuntimeError(f"No 3-month realtime-data context coverage for {zone} {target}, and ENTSOE_API_KEY is not configured.")

        # Production note: ENTSO-E actual generation split by onshore/offshore can vary by TSO.
        # Keep this isolated so the realtime data archive/backfill code can substitute curated actuals.
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=self.api_key)
        if target == "load":
            actual = client.query_load(zone_code, start=start, end=end)
            tso = client.query_load_forecast(zone_code, start=start, end=end)
        else:
            psr_type = {"solar": "B16", "wind_onshore": "B19", "wind_offshore": "B18"}[target]
            actual = client.query_generation(zone_code, start=start, end=end, psr_type=psr_type)
            tso = client.query_wind_and_solar_forecast(zone_code, start=start, end=end, psr_type=psr_type)
        actual_frame = _entsoe_response_to_hourly_frame(actual, "actual_mw")
        tso_frame = _entsoe_response_to_hourly_frame(tso, "tso_forecast_mw")
        frame = actual_frame.merge(tso_frame, on="timestamp", how="inner")
        return frame[["timestamp", "actual_mw", "tso_forecast_mw"]]

    def _stored_tso_forecast(
        self,
        *,
        run_date: date,
        zone: str,
        target: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        horizon_hours: int,
    ) -> pd.DataFrame:
        path = ForecastStore().forecast_path(run_date.isoformat())
        if not path.exists():
            return pd.DataFrame(columns=["timestamp", "tso_forecast_mw"])
        try:
            frame = pd.read_csv(path, usecols=["zone", "target", "timestamp", "tso_forecast_mw"], parse_dates=["timestamp"])
        except (ValueError, OSError):
            return pd.DataFrame(columns=["timestamp", "tso_forecast_mw"])
        frame = frame[(frame["zone"].eq(zone)) & (frame["target"].eq(target))].copy()
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "tso_forecast_mw"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["tso_forecast_mw"] = pd.to_numeric(frame["tso_forecast_mw"], errors="coerce")
        frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
        frame = frame.dropna(subset=["tso_forecast_mw"]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        if len(frame) < horizon_hours:
            return pd.DataFrame(columns=["timestamp", "tso_forecast_mw"])
        return frame[["timestamp", "tso_forecast_mw"]]

    def _stored_actual_context(self, *, zone: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = self._stored_forecasts()
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "actual_mw", "tso_forecast_mw"])
        frame = frame[(frame["zone"].eq(zone)) & (frame["target"].eq(target))].copy()
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "actual_mw", "tso_forecast_mw"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame[frame["timestamp"].between(start, end, inclusive="left")]
        for column in ["actual_mw", "tso_forecast_mw"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["actual_mw", "tso_forecast_mw"])
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        return frame[["timestamp", "actual_mw", "tso_forecast_mw"]]

    def _stored_forecasts(self) -> pd.DataFrame:
        global _STORED_FORECAST_CACHE
        if _STORED_FORECAST_CACHE is not None:
            return _STORED_FORECAST_CACHE.copy()
        store = ForecastStore()
        files = sorted((store.data_dir / "forecasts").glob("forecasts_*.csv"))
        frames = []
        for path in files:
            try:
                frame = pd.read_csv(
                    path,
                    usecols=["zone", "target", "timestamp", "actual_mw", "tso_forecast_mw"],
                    parse_dates=["timestamp"],
                )
            except (ValueError, OSError):
                continue
            frames.append(frame)
        _STORED_FORECAST_CACHE = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return _STORED_FORECAST_CACHE.copy()

    def fetch_actuals(self, *, zone: str, zone_code: str, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # This method is used for display/backfill actuals. The dashboard should
        # prefer the realtime-data snapshot archive when available, then fall back
        # to a direct ENTSO-E API call.
        archive = self.archive.fetch_actuals(zone=zone, target=target, start=start, end=end)
        if not archive.empty:
            return archive
        if not self.api_key:
            raise RuntimeError("No realtime-data archive coverage for this time window, and ENTSOE_API_KEY is not configured for direct fetch.")

        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=self.api_key)
        start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        if target == "load":
            actual = client.query_load(zone_code, start=start, end=end)
        else:
            psr_type = {"solar": "B16", "wind_onshore": "B19", "wind_offshore": "B18"}[target]
            actual = client.query_generation(zone_code, start=start, end=end, psr_type=psr_type)
        return _entsoe_response_to_hourly_frame(actual, "actual_mw").dropna()
