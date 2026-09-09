from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from time import perf_counter

import pandas as pd

from rt_forecast_dashboard.pipeline.check_forecast_complete import _requested_models, is_forecast_complete
from rt_forecast_dashboard.pipeline.run_daily import run_daily_forecast
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import BRUSSELS_TZ, latest_complete_run_date


def _date_range(start: date, end: date) -> list[date]:
    if start > end:
        return []
    return [day.date() for day in pd.date_range(start=start, end=end, freq="D")]


def _recent_start(end: date, recent_days: int) -> date:
    return end - timedelta(days=max(recent_days - 1, 0))


def _completion_status(store: ForecastStore, run_date: date, model_keys: set[str] | None) -> tuple[bool, str]:
    return is_forecast_complete(store.forecast_path(run_date.isoformat()), model_keys)


def run_backfill(start: str, end: str, model_keys: set[str] | None = None, *, force: bool = False) -> pd.DataFrame:
    store = ForecastStore()
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    days = _date_range(start_date, end_date)
    rows = []
    for day in days:
        run_date = day
        started = perf_counter()
        before_complete, before_message = _completion_status(store, run_date, model_keys)
        if before_complete and not force:
            rows.append(
                {
                    "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "run_date": run_date.isoformat(),
                    "rows": 0,
                    "seconds": round(perf_counter() - started, 3),
                    "status": "skipped_complete",
                    "message": before_message,
                }
            )
            print(f"Skip {run_date}: {before_message}")
            continue

        try:
            result = run_daily_forecast(run_date, model_keys=model_keys)
            after_complete, after_message = _completion_status(store, run_date, model_keys)
            status = "complete" if after_complete else "incomplete"
            if not after_complete:
                print(f"::warning::Forecast remains incomplete for {run_date}: {after_message}")
            rows.append(
                {
                    "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "run_date": run_date.isoformat(),
                    "rows": len(result),
                    "seconds": round(perf_counter() - started, 3),
                    "status": status,
                    "message": after_message,
                }
            )
        except Exception as exc:
            store.log_issue(run_date=run_date.isoformat(), zone="ALL", target="ALL", stage="backfill", message=str(exc))
            rows.append(
                {
                    "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "run_date": run_date.isoformat(),
                    "rows": 0,
                    "seconds": round(perf_counter() - started, 3),
                    "status": "failed",
                    "message": str(exc),
                }
            )
            print(f"::warning::Forecast failed for {run_date}: {exc}")
    report = pd.DataFrame(rows)
    out = store.data_dir / "backfill" / f"backfill_{start}_{end}.csv"
    latest = store.data_dir / "backfill" / "backfill_latest.csv"
    report.to_csv(out, index=False)
    report.to_csv(latest, index=False)
    print(f"Wrote backfill report to {out}")
    if not report.empty:
        print(report[["run_date", "status", "rows", "message"]].to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missed forecast runs.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--recent-days", type=int, default=None, help="Audit this many recent Brussels run dates ending at --end or today.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys to run.")
    parser.add_argument("--force", action="store_true", help="Rerun dates even when the existing forecast file is already complete.")
    args = parser.parse_args()

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = latest_complete_run_date(datetime.now(BRUSSELS_TZ))
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    elif args.recent_days:
        start_date = _recent_start(end_date, args.recent_days)
    else:
        raise SystemExit("Provide --start or --recent-days.")

    model_keys = _requested_models(args.models)
    run_backfill(start_date.isoformat(), end_date.isoformat(), model_keys=model_keys, force=args.force)


if __name__ == "__main__":
    main()
