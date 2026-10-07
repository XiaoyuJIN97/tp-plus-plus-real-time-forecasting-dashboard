from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MAX_IMPUTED_HOURS = 6
MAX_BOUNDARY_FILL_HOURS = 2


class ContextDataQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextQualityReport:
    quality: str
    missing_hours: int
    imputed_hours: int
    methods: str
    records: tuple[dict[str, object], ...]


def repair_context(
    frame: pd.DataFrame,
    *,
    expected_end: pd.Timestamp,
    context_hours: int,
) -> tuple[pd.DataFrame, ContextQualityReport]:
    """Restore isolated hourly context gaps and return an auditable report."""
    expected_end = pd.Timestamp(expected_end).tz_convert("UTC")
    expected_index = pd.date_range(
        expected_end - pd.Timedelta(hours=context_hours - 1),
        expected_end,
        freq="h",
    )
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    work = work.reindex(expected_index)
    work.index.name = "timestamp"

    actual = pd.to_numeric(work.get("actual_mw"), errors="coerce")
    missing = actual.isna()
    missing_count = int(missing.sum())
    if missing_count == 0:
        return work.reset_index(), ContextQualityReport("complete", 0, 0, "none", ())
    if missing_count > MAX_IMPUTED_HOURS:
        raise ContextDataQualityError(
            f"context has {missing_count} missing actual hours; maximum automatic imputation is {MAX_IMPUTED_HOURS}"
        )

    tso = pd.to_numeric(work.get("tso_forecast_mw"), errors="coerce")
    tso = tso.interpolate(method="time", limit_direction="both")
    repaired = actual.copy()
    methods: dict[pd.Timestamp, str] = {}

    isolated = missing & ~missing.shift(1, fill_value=False) & ~missing.shift(-1, fill_value=False)
    internal = isolated & repaired.ffill().notna() & repaired.bfill().notna()
    interpolated = repaired.interpolate(method="time", limit_area="inside")
    for timestamp in repaired.index[internal & interpolated.notna()]:
        repaired.loc[timestamp] = interpolated.loc[timestamp]
        methods[timestamp] = "linear_interpolation"

    remaining = repaired.isna()
    trailing = []
    for timestamp in reversed(repaired.index):
        if not remaining.loc[timestamp]:
            break
        trailing.append(timestamp)
    if 0 < len(trailing) <= MAX_BOUNDARY_FILL_HOURS:
        filled = repaired.ffill()
        for timestamp in trailing:
            if pd.notna(filled.loc[timestamp]):
                repaired.loc[timestamp] = filled.loc[timestamp]
                methods[timestamp] = "forward_fill"

    remaining = repaired.isna()
    if remaining.any():
        paired = pd.DataFrame({"actual": actual, "tso": tso}).dropna().tail(168)
        if paired.empty or tso[remaining].isna().any():
            raise ContextDataQualityError("missing context cannot be recovered from interpolation, forward fill, or TSO")
        bias = float((paired["actual"] - paired["tso"]).median())
        for timestamp in repaired.index[remaining]:
            repaired.loc[timestamp] = max(0.0, float(tso.loc[timestamp] + bias))
            methods[timestamp] = "bias_adjusted_tso"

    work["actual_mw"] = repaired
    work["tso_forecast_mw"] = tso
    records = tuple(
        {
            "timestamp_utc": timestamp.isoformat(),
            "method": methods[timestamp],
            "imputed_actual_mw": float(repaired.loc[timestamp]),
        }
        for timestamp in sorted(methods)
    )
    method_list = sorted(set(methods.values()))
    quality = "interpolated" if set(method_list) <= {"linear_interpolation", "forward_fill"} else "degraded"
    return work.reset_index(), ContextQualityReport(
        quality=quality,
        missing_hours=missing_count,
        imputed_hours=len(records),
        methods="+".join(method_list),
        records=records,
    )
