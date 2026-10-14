from __future__ import annotations

import pandas as pd
import pytest

from rt_forecast_dashboard.context_quality import ContextDataQualityError, repair_context


def _context(hours: int = 12) -> pd.DataFrame:
    timestamp = pd.date_range("2026-10-05T04:00:00Z", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "actual_mw": [100.0 + value for value in range(hours)],
            "tso_forecast_mw": [105.0 + value for value in range(hours)],
        }
    )


def test_internal_gap_is_interpolated_and_recorded() -> None:
    frame = _context().drop(index=11 - 0 - 1).reset_index(drop=True)  # 14:00 UTC

    repaired, report = repair_context(
        frame,
        expected_end=pd.Timestamp("2026-10-05T15:00:00Z"),
        context_hours=12,
    )

    row = repaired[repaired["timestamp"].eq(pd.Timestamp("2026-10-05T14:00:00Z"))].iloc[0]
    assert row["actual_mw"] == pytest.approx(110.0)
    assert report.quality == "interpolated"
    assert report.imputed_hours == 1
    assert report.records[0]["method"] == "linear_interpolation"


def test_two_missing_boundary_hours_are_forward_filled() -> None:
    frame = _context().iloc[:-2]

    repaired, report = repair_context(
        frame,
        expected_end=pd.Timestamp("2026-10-05T15:00:00Z"),
        context_hours=12,
    )

    assert repaired.tail(2)["actual_mw"].tolist() == [109.0, 109.0]
    assert report.methods == "forward_fill"


def test_larger_gap_uses_bias_adjusted_tso() -> None:
    frame = _context().drop(index=[3, 4, 5, 6]).reset_index(drop=True)

    repaired, report = repair_context(
        frame,
        expected_end=pd.Timestamp("2026-10-05T15:00:00Z"),
        context_hours=12,
    )

    assert repaired["actual_mw"].notna().all()
    assert report.imputed_hours == 4
    assert report.methods == "synthetic_tso_block"
    assert report.quality == "synthetic_block"


def test_many_scattered_missing_hours_use_bias_adjusted_tso() -> None:
    missing_index = [index for start in range(1, 60, 6) for index in (start, start + 1)]
    frame = _context(hours=60).drop(index=missing_index).reset_index(drop=True)

    repaired, report = repair_context(
        frame,
        expected_end=pd.Timestamp("2026-10-07T15:00:00Z"),
        context_hours=60,
    )

    assert repaired["actual_mw"].notna().all()
    assert report.imputed_hours == 20
    assert "bias_adjusted_tso" in report.methods
    assert report.quality == "synthetic_tso"


def test_contiguous_missing_block_is_filled_and_marked() -> None:
    frame = _context().drop(index=list(range(7))).reset_index(drop=True)

    repaired, report = repair_context(
        frame,
        expected_end=pd.Timestamp("2026-10-05T15:00:00Z"),
        context_hours=12,
    )

    assert repaired["actual_mw"].notna().all()
    assert report.imputed_hours == 7
    assert report.methods == "synthetic_tso_block"
    assert report.quality == "synthetic_block"
    assert all(record["method"] == "synthetic_tso_block" for record in report.records)


def test_more_than_total_imputation_limit_is_rejected() -> None:
    frame = _context(hours=120).drop(index=list(range(1, 120, 2))).reset_index(drop=True)

    with pytest.raises(ContextDataQualityError, match="maximum automatic imputation"):
        repair_context(
            frame,
            expected_end=pd.Timestamp("2026-10-10T03:00:00Z"),
            context_hours=120,
        )
