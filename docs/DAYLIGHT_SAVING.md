# Brussels daylight-saving transitions

Daily forecasts cover 18:00 Europe/Brussels on `run_date` up to, but excluding,
18:00 the next day. Subtract those cutoffs as **UTC instants**: a local day is
not always 24 elapsed hours. Inference normally starts at 18:30 local time.

| Run date | UTC start | UTC end (exclusive) | Hours |
| --- | --- | --- | --- |
| 2026-10-24 | 2026-10-24 16:00Z | 2026-10-25 17:00Z | 25 |
| 2026-10-25 | 2026-10-25 17:00Z | 2026-10-26 17:00Z | 24 |
| 2026-03-28 | 2026-03-28 17:00Z | 2026-03-29 16:00Z | 23 |

`forecast_horizon_hours` is shared by inference, completeness, replay, ensembles,
online accuracy, conformal calibration and onboarding. Configuration's 24 hours
denotes the nominal day-ahead window; actual requests have 23, 24 or 25 hours.
Context remains 2,208 elapsed hours. Adapters return `len(features)` predictions;
replay batches group equal-length windows. Never copy/interpolate a prediction
to manufacture the extra hour.

Hour 25 shares the final learned-ensemble bucket (hours 19–24). Conformal bands
pool the final bucket's available errors from fully observed earlier runs,
accommodating mixed-length histories without stacking unequal arrays or needing
decades of autumn-only calibration samples.

All joins, deduplication and ordering use UTC. The autumn hours are both retained:

```text
2026-10-25 00:00Z → 2026-10-25 02:00 CEST (UTC+0200)
2026-10-25 01:00Z → 2026-10-25 02:00 CET  (UTC+0100)
```

Plots use continuous UTC coordinates with explicit Brussels tick/hover labels.
A single compact transition tick shows `02:00 CEST → 02:00 CET`; hover text
identifies each observation separately. Ordinary ticks keep a minimum spacing,
so there are no crowded ambiguous labels or collapsed/reversed points. Duplicate UTC rows are
removed per model/zone/target, retaining the latest run. In spring, nonexistent
local 02:00 is not inserted; the UTC line remains continuous.

The daily schedule already has UTC triggers guarded by the Brussels clock:
18:30 is 16:30 UTC in summer and 17:30 UTC in winter. No manual cron edit is
necessary. Collector requests and timestamps must remain UTC.

## Operator checks

1. On 24 October, verify 25 distinct consecutive UTC timestamps per model group,
   ending at 25 October 16:00Z. The next run starts at 17:00Z with 24 rows.
   There must be no gap or overlap.
2. Run `check_forecast_complete --date YYYY-MM-DD` for each inference pass;
   check optional candidate models explicitly in their own jobs.
3. Verify ensemble rows, observed-run scores and probabilistic panels retain
   the full window, including the extra prediction and its calibrated interval.
4. Inspect deterministic and probabilistic plots: `02:00 CEST` precedes
   `02:00 CET`, at separate instants, with no duplicate UTC point.
5. Record/recover failures if a model rejects the horizon or inputs are missing.
   Do not fabricate an hour or trim a window to pass validation.

Previously saved fixed-24-hour transition runs are not silently rewritten or
certified complete. Re-run using genuine archived inputs and preserve replay
provenance if repair is needed. The changed onboarding policy/source invalidates
admission fingerprints; regenerate evidence before admitting a new model.

Regression tests: `tests/test_dst_windows.py`. The foundation integration workflow
also exercises native CPU forecasts at 23, 24 and 25 hours for T0 Gamma and
Timestral. Local stub/demo checks alone do not prove native inference.
