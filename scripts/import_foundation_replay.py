"""Append validated mock predictions and refresh common realized outcomes.

Existing predictions are hashed before and after; only their realized outcomes
may change when the collector has revised them. Run on the production checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rt_forecast_dashboard.storage import ForecastStore

KEYS = ["run_date", "zone", "target", "model", "timestamp"]


def prediction_hash(frame):
    selected = frame[KEYS + ["forecast_mw"]].copy()
    selected["timestamp"] = pd.to_datetime(selected.timestamp, utc=True).astype(str)
    selected["run_date"] = selected.run_date.astype(str)
    selected = selected.sort_values(KEYS).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(selected, index=False).values.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    replay = pd.concat([pd.read_csv(p) for p in sorted((args.replay / "forecasts").glob("*.csv"))], ignore_index=True)
    replay["timestamp"] = pd.to_datetime(replay.timestamp, utc=True)
    if not replay.source.eq("historical_mock_replay").all():
        raise ValueError("Unlabeled historical forecasts")
    actual_keys = ["zone", "target", "timestamp"]
    actuals = replay[actual_keys + ["actual_mw"]].dropna()
    if (actuals.groupby(actual_keys).actual_mw.nunique() > 1).any():
        raise ValueError("Inconsistent realized actuals between replay models")
    actuals = actuals.drop_duplicates(actual_keys).rename(columns={"actual_mw": "refreshed_actual_mw"})
    store = ForecastStore(args.data_dir)
    baseline = store.read_forecasts()
    old_models = ~baseline.model.isin(["t0_gamma_online", "timestral_online"])
    old_hash = prediction_hash(baseline[old_models])
    audit = []
    for run_date, incoming in replay.groupby("run_date"):
        path = store.forecast_path(str(run_date))
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not existing.empty:
            existing["timestamp"] = pd.to_datetime(existing.timestamp, utc=True)
            existing = existing.merge(actuals, on=actual_keys, how="left", validate="many_to_one")
            previous = pd.to_numeric(existing.actual_mw, errors="coerce")
            fresh = existing.pop("refreshed_actual_mw")
            changed = fresh.notna() & previous.notna() & ~np.isclose(fresh, previous, atol=0.01, rtol=0)
            filled = fresh.notna() & previous.isna()
            audit.append(dict(run_date=str(run_date), revised_rows=int(changed.sum()), filled_rows=int(filled.sum())))
            existing["actual_mw"] = fresh.combine_first(previous)
            existing.to_csv(path, index=False)
        store.append_forecasts(incoming, str(run_date))
    after = store.read_forecasts()
    after = after[~after.model.isin(["t0_gamma_online", "timestral_online"])]
    # Round-trip floating point serialization can differ in the last bit;
    # compare each original prediction explicitly as well as recording hashes.
    matched = baseline[old_models][KEYS + ["forecast_mw"]].merge(after[KEYS + ["forecast_mw"]], on=KEYS, suffixes=("_before", "_after"), validate="one_to_one", how="outer", indicator=True)
    if not matched._merge.eq("both").all() or not np.allclose(matched.forecast_mw_before, matched.forecast_mw_after, atol=1e-8, rtol=1e-12, equal_nan=True):
        raise ValueError("Existing forecast predictions changed during import")
    pd.DataFrame(audit).to_csv(args.replay / "actual_refresh_audit.csv", index=False)
    (args.replay / "import_validation.json").write_text(json.dumps(dict(new_rows=len(replay), original_forecast_rows=int(old_models.sum()), predictions_preserved=True, absolute_tolerance_mw=1e-8, relative_tolerance=1e-12, original_prediction_hash=old_hash, resulting_prediction_hash=prediction_hash(after), revised_actual_rows=sum(x["revised_rows"] for x in audit), newly_realized_rows=sum(x["filled_rows"] for x in audit)), indent=2))
    print("Imported", len(replay), "mock rows; original predictions preserved")


if __name__ == "__main__":
    main()
