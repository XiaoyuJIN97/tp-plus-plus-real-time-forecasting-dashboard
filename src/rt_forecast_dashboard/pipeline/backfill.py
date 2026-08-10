from __future__ import annotations

import argparse
from datetime import datetime
from time import perf_counter

import pandas as pd

from rt_forecast_dashboard.pipeline.run_daily import run_daily_forecast
from rt_forecast_dashboard.storage import ForecastStore


def run_backfill(start: str, end: str, model_keys: set[str] | None = None) -> None:
    store = ForecastStore()
    days = pd.date_range(start=start, end=end, freq="D")
    rows = []
    for day in days:
        run_date = day.date()
        started = perf_counter()
        try:
            result = run_daily_forecast(run_date, model_keys=model_keys)
            rows.append(
                {
                    "run_date": run_date.isoformat(),
                    "rows": len(result),
                    "seconds": round(perf_counter() - started, 3),
                    "status": "ok",
                    "message": "",
                }
            )
        except Exception as exc:
            store.log_issue(run_date=run_date.isoformat(), zone="ALL", target="ALL", stage="backfill", message=str(exc))
            rows.append(
                {
                    "run_date": run_date.isoformat(),
                    "rows": 0,
                    "seconds": round(perf_counter() - started, 3),
                    "status": "failed",
                    "message": str(exc),
                }
            )
    out = store.data_dir / "backfill" / f"backfill_{start}_{end}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote backfill report to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missed forecast runs.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys to run.")
    args = parser.parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    model_keys = {value.strip() for value in args.models.split(",")} if args.models else None
    run_backfill(args.start, args.end, model_keys=model_keys)


if __name__ == "__main__":
    main()
