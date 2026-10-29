from __future__ import annotations

from datetime import date

import numpy as np
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
        # Generation and plant consumption are different measures, not assets
        # to add together. ENTSO-E drops this column level in months with only
        # one measure, so accept both shapes before stitching monthly windows.
        aggregated = [column for column in numeric.columns
                      if "Actual Aggregated" in (column if isinstance(column, tuple) else (column,))]
        consumption = [column for column in numeric.columns
                       if "Actual Consumption" in (column if isinstance(column, tuple) else (column,))]
        if aggregated:
            numeric = numeric.loc[:, aggregated]
        elif consumption:
            numeric = numeric.drop(columns=consumption)
        if numeric.empty:
            raise ValueError("ENTSO-E response has no numeric generation columns")
        values = numeric.sum(axis=1, min_count=1)
    else:
        values = response
    values = values.rename(value_col)
    if isinstance(values.index, pd.MultiIndex):
        date_level = next(
            (
                index
                for index, level in enumerate(values.index.levels)
                if pd.api.types.is_datetime64_any_dtype(level.dtype)
            ),
            None,
        )
        if date_level is not None:
            values.index = values.index.get_level_values(date_level)
    frame = values.reset_index().rename(columns={"index": "timestamp"})
    timestamp_col = "timestamp" if "timestamp" in frame.columns else next(
        (column for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column])),
        frame.columns[0],
    )
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


def _query_entsoe_hourly(query, value_col: str, *, start: pd.Timestamp,
                         end: pd.Timestamp, **kwargs) -> pd.DataFrame:
    """Normalize each month before entsoe-py can concatenate mixed headers.

    Version 0.8.1 splits long calls internally and concatenates raw responses.
    A single-measure month has flat columns; a generation/consumption month
    has MultiIndex columns. Combining those fails on pandas 3 before our usual
    response converter is reached. Limit each public API call to one month.
    """
    from entsoe.exceptions import NoMatchingDataError

    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    frames = []
    cursor = start
    while cursor < end:
        stop = min(cursor + pd.DateOffset(months=1), end)
        try:
            response = query(start=cursor, end=stop, **kwargs)
        except NoMatchingDataError:
            cursor = stop
            continue
        frame = _entsoe_response_to_hourly_frame(response, value_col)
        frames.append(frame[frame.timestamp.ge(cursor) & frame.timestamp.lt(stop)])
        cursor = stop
    if not frames:
        raise NoMatchingDataError("No ENTSO-E observations in requested interval")
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


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
        archive = _forecast_on_grid(archive, start, horizon_hours)
        if not archive.empty:
            return archive
        stored = self._stored_tso_forecast(run_date=run_date, zone=zone, target=target, start=start, end=end, horizon_hours=horizon_hours)
        stored = _forecast_on_grid(stored, start, horizon_hours)
        if not stored.empty:
            return stored
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
        frame = _forecast_on_grid(_entsoe_response_to_hourly_frame(series, "tso_forecast_mw"), start, horizon_hours)
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

        return self.fetch_direct_actual_context(run_date=run_date, zone_code=zone_code, target=target, context_hours=context_hours)

    def fetch_direct_actual_context(self, *, run_date: date, zone_code: str, target: str, context_hours: int) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame(columns=["timestamp", "actual_mw", "tso_forecast_mw"])
        end = brussels_cutoff_timestamp(run_date)
        start = end - pd.Timedelta(hours=context_hours + 72)
        # Production note: ENTSO-E actual generation split by onshore/offshore can vary by TSO.
        # Keep this isolated so the realtime data archive/backfill code can substitute curated actuals.
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=self.api_key)
        if target == "load":
            actual_query, forecast_query = client.query_load, client.query_load_forecast
            kwargs = {}
        else:
            psr_type = {"solar": "B16", "wind_onshore": "B19", "wind_offshore": "B18"}[target]
            actual_query, forecast_query = client.query_generation, client.query_wind_and_solar_forecast
            kwargs = {"psr_type": psr_type}
        actual_frame = _query_entsoe_hourly(actual_query, "actual_mw", country_code=zone_code,
                                           start=start, end=end, **kwargs)
        tso_frame = _query_entsoe_hourly(forecast_query, "tso_forecast_mw", country_code=zone_code,
                                        start=start, end=end, **kwargs)
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
            query, kwargs = client.query_load, {}
        else:
            psr_type = {"solar": "B16", "wind_onshore": "B19", "wind_offshore": "B18"}[target]
            query, kwargs = client.query_generation, {"psr_type": psr_type}
        return _query_entsoe_hourly(query, "actual_mw", country_code=zone_code,
                                    start=start, end=end, **kwargs).dropna()


def _forecast_on_grid(frame: pd.DataFrame, start: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Never accept an API calendar-day response by row count alone."""
    if frame.empty:
        return frame
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.drop_duplicates('timestamp', keep='last').set_index('timestamp')
    frame = frame.reindex(pd.date_range(start, periods=hours, freq='h', name='timestamp'))
    values = pd.to_numeric(frame.tso_forecast_mw, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        return frame.iloc[:0].reset_index()
    return frame.reset_index()
