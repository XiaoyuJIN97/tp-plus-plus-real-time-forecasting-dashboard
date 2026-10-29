"""Audit mock replay coverage and compare against saved forecasts on exact keys."""
from rt_forecast_dashboard.time_utils import forecast_horizon_hours, brussels_cutoff_timestamp
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["run_date", "zone", "target", "timestamp"]


def matched_complete_windows(frame):
    valid=[]
    for (day,zone,target),part in frame.groupby(['run_date','zone','target']):
        origin=brussels_cutoff_timestamp(pd.Timestamp(day).date())
        grid=pd.date_range(origin,periods=forecast_horizon_hours(day),freq='h')
        if (pd.DatetimeIndex(part.timestamp.sort_values()).equals(grid)
                and pd.to_datetime(part.context_end,utc=True).eq(origin-pd.Timedelta(hours=1)).all()
                and pd.to_datetime(part.context_end_baseline,utc=True).eq(origin-pd.Timedelta(hours=1)).all()):
            valid.append(part)
    return pd.concat(valid,ignore_index=True) if valid else frame.iloc[:0].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()
    new = pd.concat([pd.read_csv(p) for p in sorted((args.replay / "forecasts").glob("*.csv"))], ignore_index=True)
    old = pd.concat([pd.read_csv(p) for p in sorted((args.history / "forecasts").glob("*.csv"))], ignore_index=True)
    compact = args.history / "dashboard/forecasts_recent.parquet"
    if compact.exists():
        old = pd.concat([old, pd.read_parquet(compact)], ignore_index=True)
    for frame in (old, new):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        frame["run_date"] = frame.run_date.astype(str)
    old = old.drop_duplicates(KEYS + ["model"], keep="last")
    if new.duplicated(KEYS + ["model"]).any() or not np.isfinite(new.forecast_mw).all() or (new.forecast_mw < 0).any():
        raise ValueError("Duplicate or invalid replay forecasts")
    counts = new.groupby(["run_date", "zone", "target", "model"]).timestamp.nunique()
    if not all(count == forecast_horizon_hours(key[0]) for key,count in counts.items()):
        raise ValueError("Incomplete saved replay horizons")
    for key, part in new.groupby(['run_date','zone','target','model']):
        grid = pd.date_range(brussels_cutoff_timestamp(pd.Timestamp(key[0]).date()), periods=forecast_horizon_hours(key[0]), freq='h')
        if not pd.DatetimeIndex(part.timestamp.sort_values()).equals(grid):
            raise ValueError('Replay is outside the exact midnight-ending delivery grid')
    context_end = pd.to_datetime(new.context_end, utc=True)
    first = new.groupby(["run_date", "zone", "target", "model"]).timestamp.transform("min")
    if not context_end.eq(first - pd.Timedelta(hours=1)).all():
        raise ValueError("Replay context does not end before forecast")
    coverage = new.groupby(["zone", "target", "model"], as_index=False).agg(
        first_run=("run_date", "min"), last_run=("run_date", "max"),
        windows=("run_date", "nunique"), rows=("timestamp", "size"), actual_rows=("actual_mw", "count"))
    coverage.to_csv(args.replay / "coverage.csv", index=False)
    metrics = []
    for (zone, target, model), group in new.groupby(["zone", "target", "model"]):
        baselines = old[(old.zone == zone) & (old.target == target)]
        for base, reference in baselines.groupby("model"):
            if base in {"tabpfn_online", "t0_gamma_online", "timestral_online"}:
                continue
            matched = group[KEYS + ["forecast_mw", "actual_mw", "context_end"]].merge(
                reference[KEYS + ["forecast_mw", "context_end"]], on=KEYS, suffixes=("", "_baseline"), validate="one_to_one")
            matched = matched.dropna(subset=["actual_mw", "forecast_mw", "forecast_mw_baseline"])
            matched = matched_complete_windows(matched)
            if matched.empty:
                continue
            error = matched.forecast_mw - matched.actual_mw
            baseline_error = matched.forecast_mw_baseline - matched.actual_mw
            mae, base_mae = error.abs().mean(), baseline_error.abs().mean()
            origins = pd.to_datetime(matched.context_end, utc=True).eq(pd.to_datetime(matched.context_end_baseline, utc=True))
            metrics.append(dict(same_origin_rows=int(origins.sum()), origin_aligned=bool(origins.all()), zone=zone, target=target, model=model, baseline=base,
                                n=len(matched), windows=matched.run_date.nunique(),
                                mae=mae, baseline_mae=base_mae,
                                mae_change_pct=100 * (mae / base_mae - 1) if base_mae else np.nan,
                                rmse=np.sqrt((error**2).mean()), baseline_rmse=np.sqrt((baseline_error**2).mean())))
    scores = pd.DataFrame(metrics,columns=['same_origin_rows','origin_aligned','zone','target','model','baseline',
        'n','windows','mae','baseline_mae','mae_change_pct','rmse','baseline_rmse'])
    scores.to_csv(args.replay / "matched_comparison.csv", index=False)
    tfm = scores[scores.baseline.eq("timesfm3_online")]
    lines = ["# T0 Gamma and Timestral online mock replay", "",
             f"Mock forecasts cover {new.run_date.min()} through {new.run_date.max()}, with 2,208 context hours and 29/30/31 forecast hours through next-day midnight. Actual context ends one hour before delivery.", "",
             "All rows are labeled `historical_mock_replay`. Collector archives and retained outcomes reconstruct past inputs. October weather gaps use the Open-Meteo historical forecast API; these are not certified original issue-time vintages. This is a reconstructed mock comparison, not a claim of contemporaneous live performance.", "",
             "Coverage gaps remain excluded under production context-quality limits. Comparisons require identical complete midnight-ending delivery grids, the same forecast origin and the same observed actuals. Old 24-hour or misaligned baseline windows are excluded.", "",
             f"Saved predictions: **{len(new):,} rows / {len(counts):,} complete daily windows** across {new.zone.nunique()} zones.", "",
             "## Matched comparison with TimesFM3", "",
             "| Zone | Target | Model | Scored hours | MAE (MW) | TimesFM3 MAE | Change |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in tfm.itertuples():
        lines.append(f"| {row.zone} | {row.target} | {row.model.replace('_online', '')} | {row.n} | {row.mae:.2f} | {row.baseline_mae:.2f} | {row.mae_change_pct:+.1f}% |")
    lines += ["", "See `coverage.csv` for individual model coverage, `replay_issues.csv` for excluded windows, and `matched_comparison.csv` for comparisons with the other saved models. Negative changes indicate lower MAE."]
    (args.replay / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.replay / "validation.json").write_text(json.dumps(dict(rows=len(new), windows=len(counts), finite=True, complete_horizons=True, duplicate_free=True, context_before_delivery=True), indent=2))
    print(tfm.groupby("model").agg(series=("target", "size"), wins=("mae_change_pct", lambda s: int((s < 0).sum())), median_change=("mae_change_pct", "median")).to_string())


if __name__ == "__main__":
    main()
