# DK solar full-horizon history repair — 3 November 2026

The earlier DK solar mocks existed, but 735 base-model groups still used the
former 24-hour window. Their full-window replays had failed the existing context
quality limit: archived 92-day contexts contained more than 48 missing actual
hours. The full-window leaderboard correctly excluded these legacy groups,
which shortened DK1 solar scoring to 21 September and DK2 to 20 September.
The original migration was therefore incomplete for this scope.

## Published result

| Series | Before: shared point runs | After: shared point runs | Retained point period | Interval-score runs after |
|---|---:|---:|---|---:|
| DK1 solar | 42 | 89 | 5 August–1 November 2026 | 65 |
| DK2 solar | 43 | 89 | 5 August–1 November 2026 | 65 |

All eight displayed models have 2,671 scored delivery intervals per area.
There are 102 complete archived origins per displayed DK solar model from
**24 July through 2 November**; the latest origin is not fully observed yet.
The dashboard's unchanged 90-day retention starts on 5 August. The interval
leaderboard retains its existing calibration warm-up, complete-window and
shared-truth checks, so it does not score all 89 point runs.

Native inference regenerated **735 base-model windows** with zero failures;
**105 median-ensemble windows** were rebuilt. Existing complete component
predictions were preserved. Validation verified **623,776 forecast rows outside
DK solar remained unchanged**, as did previously complete DK solar model
predictions. Other scopes' outcomes can refresh from the current collector
snapshot, but their model forecasts/ensemble settings were not changed.

The [corrected TSO comparison figure](tso_solar_performance_same_period.png)
uses 82 identical qualifying origin dates across all seven areas within
5 August–1 November. DK1 WAPE is 22.13% and DK2 23.40%; the earlier September-only
figures are superseded by this full-period comparison. [Metrics](tso_comparison_after.csv)
include the counts and exact dates.

See [validation.json](validation.json), [point scores](point_scores_after.csv),
[interval scores](interval_scores_after.csv) and the
[retained coverage report](../../data/dashboard/forecast_history_coverage.csv).
The raw coverage report also lists disabled NL/hidden TabPFN history; it is not
a statement that these are displayed online. Unrelated unrecovered legacy
windows are still excluded; this repair certifies the DK solar scope only.

## Observation recovery: published evidence, not assumed nighttime zeros

Re-querying ENTSO-E did not restore the missing intervals. For example, its
19–20 June DK1 response contains daylight-only periods, not an explicit zero
covering the intervening nighttime. We did not extend those periods, interpret
missing values as zero, or relax the 48-hour context limit.

Energinet's official [Electricity Production and Exchange 5 min Realtime dataset](https://www.energidataservice.dk/tso-electricity/ElectricityProdex5MinRealtime)
provides per-area `SolarPower` in MW. Its
[metadata catalog](https://api.energidataservice.dk/meta/dataset) describes
five-minute production and exchange values. The
[frozen request record](energinet_request.json) records the endpoint, dates,
columns and retrieval time; [dataset metadata](energinet_dataset_metadata.json)
is included. The frozen source has **89,196 five-minute records**.

Recovery accepts an hour only when:

- its price area is explicitly DK1 or DK2;
- all twelve distinct, correctly aligned five-minute UTC records are present;
- every value is finite and exactly zero;
- duplicate records do not conflict; and
- the archived value is missing. Existing observations always take priority.

This recovered **49 DK1 and 108 DK2 hourly values**. All are explicit publisher
zeros; no nonzero values from another series were substituted. Energinet's
settlement dataset was checked and rejected as a substitute because its
measurement/revision basis differs. Remaining context gaps use the unchanged,
audited production repair policy: maximum 25 hours per DK1 window and 2 per DK2
window in this replay. Every repaired delivery window has fully observed truth.
Context imputation is never used as scoring truth.

The [recovery helper](../../src/rt_forecast_dashboard/data/energinet_solar_recovery.py)
and [hourly evidence ledger](recovered_published_zero_actuals.parquet) make the
rule reproducible. Forecast truth for repaired DK solar origins is attached
consistently across models from the frozen archived series plus this evidence.

## Forecast settings and replay parity

The deployed checkpoints, four weather points, covariates, CPU execution,
2,208-hour context and 18:00 Brussels-to-next-day-midnight grid are unchanged.
Native model calls use equal-length batches and return all delivery hours;
no forecast tail is padded, copied from another origin or inferred from truth.

The audit found one batch/single-adapter parity bug: Chronos batch replay did
not apply the online adapter's nonnegative clipping. The batch helper now uses
the same clipping, with a regression test. Twelve raw negative Chronos outputs
were clipped to zero as the online adapter already does; original values are
recorded in [the clipping ledger](chronos_online_clipping_ledger.csv).

These are **historical mock replays from revised archived inputs**, including
weather restored previously. They are not certified as-issued forecast vintages
or a purely prospective trial. Point-source provenance is retained. The original
pre-repair rows are archived in `original_dk_solar_forecasts.parquet`.

## Reproduce or perform a future controlled repair

Frozen original input series, weather, source observations, native timing and
error ledgers, the explicit repair plan, context-quality/imputation ledgers and
hash manifest are included. `reproduction_inputs` contains the original archive
before zero recovery, with weather CSVs compressed as `.csv.gz`. Decompress them
in a separate staging copy, never on the original archive or PSSD.

At this repair's code revision, create a read-only source checkout at the
[pre-repair revision](https://github.com/XiaoyuJIN97/tp-plus-plus-real-time-forecasting-dashboard/tree/b7493c36b54030a1cc1f07d69d8b067fdfe6d1ec),
then prepare a new input directory:

```sh
PYTHONPATH=src python scripts/prepare_dk_solar_recovery_inputs.py \
  --source /path/to/decompressed-original-inputs \
  --observations docs/dk_solar_history_repair_2026-11-03/energinet_published_5min.parquet \
  --output /path/to/staging/dk/inputs

# Use the existing host and isolated foundation CPU environments/checkpoint cache.
PYTHONPATH=src python scripts/batch_replay_midnight.py \
  --source /path/to/pre-repair-checkout/data \
  --inputs /path/to/staging/dk/inputs \
  --plan docs/dk_solar_history_repair_2026-11-03/repair_plan.csv \
  --output /path/to/staging/dk/replay \
  --start 2026-07-24 --end 2026-09-20 --scopes DK1:solar,DK2:solar

PYTHONPATH=src python scripts/stage_dk_solar_history_repair.py \
  --source /path/to/pre-repair-checkout/data \
  --replay /path/to/staging/dk/replay \
  --plan docs/dk_solar_history_repair_2026-11-03/repair_plan.csv \
  --inputs /path/to/staging/dk/inputs \
  --output /path/to/staging/dk/publication
```

The staging importer rejects incomplete/unregistered replay groups before writing
anything, preserves already-complete predictions, restricts the repair to DK
solar and writes no source data. Rebuild affected ensembles and dashboard
artifacts against the staged store, validate exact historical grids, scored
periods and unaffected scopes, then publish reviewed results with the usual
GitHub data publisher. Frozen input recovery performs no upstream requests.
For a fresh source download, respect the public API's returned `Retry-After`
rate limit (one request per minute during this audit).

The dashboard artifact builder now emits `forecast_history_coverage.csv` with
stored/full-horizon/fully-observed run counts and first/last usable dates by
area, target and model. Operators can spot a series whose stored history starts
before its usable history without relying on the page's overall date range.
Regression tests cover recovery evidence, preservation of other forecasts,
short-window rejection, the published early DK windows, DST and Chronos clipping.
