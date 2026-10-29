# Forecasts through next-day midnight

Every production run starts at 18:00 Europe/Brussels on its run date and ends
at midnight after the following calendar day, exclusive. On an ordinary day
this is 30 hourly delivery intervals; the clock-change windows have 29 or 31.
The 23:00 timestamp denotes the 23:00–24:00 interval. Inference remains scheduled
at/after 18:30. All storage, requests, joins and completeness checks use UTC.

`time_utils.forecast_end_timestamp` and `forecast_horizon_hours` define the
contract for online inference, archived replay, offline evaluation, ensembles,
point-score tables and interval-score tables. The first six hours use the
current delivery day's TSO day-ahead product; the next 24 use the following
day's product. Do not describe all 30 hours as one newly issued TSO forecast.

Daily runs intentionally overlap by six delivery hours. Plotting retains the
latest forecast per UTC delivery instant; scoring retains distinct run origins.
Only fully observed, complete, shared model windows enter comparisons. A longer
window becomes eligible later; neither ensemble weights nor interval calibration
may use yesterday's still-unfinished outcomes. Learned weights and calibration
have five buckets: 1–6, 7–12, 13–18, 19–24, 25–30; hour 31 shares the last bucket.
Old offline weights lack the new bucket: use equal-weight warmup until enough
complete earlier runs exist, rather than extrapolating unsupported weights.

## Historical inputs and reruns

Keep the current live artifact until a separately staged replacement has been
validated. Keep the original forecast CSVs and model checkpoint revisions.
PSSD is an input archive and must remain read-only.

```sh
PYTHONPATH=src python scripts/prepare_midnight_archive.py \
  --collector /path/to/entsoe-collector \
  --pssd /Volumes/PSSD/1_entsoe-realtime-data-archive/data-branch-work/data \
  --output /path/to/staging/inputs/entsoe --start 2026-07-24 --end YYYY-MM-DD

PYTHONPATH=src python scripts/audit_midnight_windows.py \
  --source data --inputs /path/to/staging/inputs \
  --output /path/to/staging/window_audit.csv --start 2026-07-24 --end YYYY-MM-DD
```

Required input layout:

- `inputs/entsoe/data/hourly/ZONE/VARIABLE/YEAR.parquet`: `timestamp_utc`, `value`;
  actual and day-ahead-forecast variables for each enabled target. Include at
  least 2,208 hours before the earliest origin and the whole delivery window.
- `inputs/weather/data/raw/ZONE/TARGET/YEAR.csv`: `timestamp_utc`, the deployed
  point-specific weather covariates, and `deg_proxy` where required. Collector
  weather targets are `load`, `solar`, `onshore`, `offshore`. Keep at least the
  72-hour preparation margin before the context.
- Source forecast history: daily `forecasts_YYYY-MM-DD.csv` plus the compact
  parquet, if present. Those files supply the old origin/model inventory; their
  24-hour predictions do not supply the missing six forecasts.

Resolve missing inputs before rerunning. If using Open-Meteo's historical
forecast service, retrieve forecast weather, not realized weather. Restored
latest/revised snapshots must be marked retrospective; they do not prove input
availability at the original issue time. Strict onboarding availability gates
still require separate provenance evidence or the documented review process.

For efficient replay, use the same native adapters with equal-length batches:

```sh
# Host has Chronos2/TimesFM3; FOUNDATION_PYTHON points to the isolated T0/Timestral runtime.
export FOUNDATION_PYTHON=/path/to/foundation-env/bin/python
export FOUNDATION_CACHE=/path/to/checkpoint/cache
PYTHONPATH=src python scripts/batch_replay_midnight.py \
  --source data --inputs /path/to/staging/inputs --output /path/to/staging/results \
  --start 2026-07-24 --end YYYY-MM-DD --batch-size 8
```

The sequential `scripts/replay_midnight_history.py` runs the unchanged online
entry point with archive-only weather, accepts optional scope/model filters,
and is useful for diagnosis or replaying a failed window. Both replay paths
resume complete groups and record explicit failure evidence. Neither path
publishes to production. Never append forecasts copied from another origin,
repeat the last six predictions, or use future actuals as covariates.

Assemble validated groups into a separate copy of retained history. This
rejects shifted grids, invalid values or incorrect replay provenance, and writes
an explicit list of unrecovered legacy groups:

```sh
PYTHONPATH=src python scripts/assemble_midnight_history.py \
  --source data --inputs /path/to/staging/results \
  --output /path/to/staging/assembled --start 2026-07-24 --end YYYY-MM-DD
```

After all required groups pass, import full groups, recompute selected ensembles
from earlier completed runs, attach observed actuals, and rebuild dashboard
artifacts and both leaderboard tables. Retain failed/missing scopes in the
migration report; do not certify them complete or mix 24-hour reference scores
with 30-hour candidate scores.

## Offline 2024 and 2025

Use the identical delivery window and covariates, with 2023 context for early
2024 and next-year inputs for late-December windows. Generate a daily inventory:

```sh
PYTHONPATH=src python scripts/plan_midnight_offline.py \
  --output /path/to/staging/offline-plan.csv --years 2024,2025
```

The plan contains `run_date,zone,target,model,origin_utc,end_utc,hours`. It can
be restricted with `--scopes BE:load` and `--models t0_gamma_online,timestral_online`.
Normalize historical inputs to the layout above. Then run the online-equivalent
executor against the explicit plan, writing only to a new staging directory:

```sh
export ENTSOE_REALTIME_DATA=/path/to/offline-inputs/entsoe
export ENTSOE_REALTIME_RAW_BASE=''
export ENTSOE_REALTIME_GIT_REF=''
export OPEN_METEO_REALTIME_DATA=/path/to/offline-inputs/weather
export OPEN_METEO_REALTIME_GIT_REF=''
PYTHONPATH=src python scripts/replay_midnight_history.py \
  --source data --output /path/to/staging/offline-results --stage offline \
  --plan /path/to/staging/offline-plan.csv --start 2024-01-01 --end 2025-12-31
```

For admission, use the standard frozen-window exporter, isolated benchmark
runner, annual matched-reference metrics and runtime/resource gates described
in `model_onboarding/README.md`. Their policy now requires 29/30/31-hour windows.
Regenerate every reference prediction at the new horizon; old 24-hour tables and
admission fingerprints are historical evidence only. Report 2024 and 2025
separately, and measure cold and warm inference on the declared hardware.
