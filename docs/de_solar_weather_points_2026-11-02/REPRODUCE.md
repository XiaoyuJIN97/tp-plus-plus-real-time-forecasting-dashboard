# Reproduce the DE solar weather-point study

The study is offline. No collector settings, model registry, ensemble settings,
or saved dashboard forecasts are changed by these commands.

## Included data

- `preperiod_capacity_grid.parquet` and `current_capacity_grid.parquet`: coarse
  capacity-weighted public geography derived from all 67 solar segments of the
  2 November MaStR export. No operator names, addresses or unit IDs are retained.
- `preperiod_points.csv`: proposed four-point input configuration. The current
  fleet selected exactly the same snapped coordinates (`current_points.csv`).
- `reproduction_inputs/inputs/{baseline,candidate}/`: frozen common ENTSO-E
  actual/TSO history plus the two weather series. Each weather arm covers
  23 April–1 November 2026 at hourly resolution, sufficient for the 92-day context.
- `reproduction_inputs/online_tso_by_origin.parquet`: the saved online TSO curves
  used in both arms. These preserve origin-specific overlapping delivery windows.
- `paired_predictions.parquet`: all **47,040 predictions**, including controls
  and ensembles, for 98 shared complete origins and eight model families.
- `input_hashes.json`, `configuration_hashes.json`, `runtime_environment.json`:
  input/configuration fingerprints, model runtime versions and pinned foundation
  checkpoint revisions. Native model checkpoints are not redistributed here.

Keep the existing online CPU model environments and cached weights. The host and
isolated foundation environments are described in
[the worked model examples](../model_onboarding/FOUNDATION_EXAMPLES.md).
Lightweight selection/report/test dependencies are in
[requirements-weather-study.txt](../../requirements-weather-study.txt).

## Repeat selection from the frozen aggregate

No full registry download is needed to reproduce the proposed coordinates:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 python - <<'PY'
import pandas as pd
from scripts.select_de_solar_weather_points import select_points
root = 'docs/de_solar_weather_points_2026-11-02'
points = select_points(pd.read_parquet(f'{root}/preperiod_capacity_grid.parquet'))
print(points[['point', 'gfs_lat', 'gfs_lon', 'cluster_share']])
PY
```

Capacity shares are used to select representative locations. They are not
additional covariate weights or new ensemble weights. The legacy `gfs_lat` /
`gfs_lon` column names store request coordinates; the weather request policy
remained Open-Meteo automatic best match.

The full extraction script accepts `--root`, `--url`, `--workers` and `--as-of`.
It reads only solar ZIP members using resumable byte ranges and verifies their
CRC/length. The original daily export may expire at the provider, so preserve
its compressed cache if a unit-level extraction must be repeated. The published
coarse aggregates and checksums allow selection to be reproduced independently.

## Repeat inference with the frozen data

From the repository root, using the original model configuration:

```bash
cp -R docs/de_solar_weather_points_2026-11-02/reproduction_inputs /tmp/de-solar-replay
cp docs/de_solar_weather_points_2026-11-02/*capacity_grid.parquet /tmp/de-solar-replay/
export PYTHONPATH=src:.
export FOUNDATION_PYTHON=/absolute/path/to/the/existing/foundation-env/bin/python
export FOUNDATION_CACHE=/absolute/path/to/the/existing/huggingface/cache
export FOUNDATION_DEVICE=cpu
export FOUNDATION_THREADS=4
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
python -m scripts.de_solar_weather_study run --root /tmp/de-solar-replay \
  --start 2026-07-24 --end 2026-10-31
```

A configuration hash check runs before and after inference. Four weather points,
2,208 context hours, 18:00 Europe/Brussels cutoff, the 30-hour nominal horizon
(29/31 at DST), CPU adapters, all model parameters and the DE-solar median
ensemble remain unchanged. TSO and persistence must match across arms exactly.

Generate tables and plots from the replay:

```bash
python -m scripts.report_de_solar_weather_study --root /tmp/de-solar-replay \
  --output /tmp/de-solar-replay-report
```

For a new input recovery, run the `prepare` subcommand with a staged ENTSO-E input
root and the saved online forecast directory, then the `weather` subcommand for
each point file. Historical weather uses the same API, variables and retrieval
method for both arms. See `python -m scripts.de_solar_weather_study --help`.

## Evaluation boundaries

The 100 planned origins are 24 July–31 October. Origins **23 and 24 October**
lack complete saved online TSO curves and are excluded from both arms. The
98 eligible origins each have 30 delivery hours: **2,940 rows per model**.
The DST origin would require 31 values and is excluded by the same exact-grid
check, not shortened to 30. The latest scored delivery is 1 November at 23:00
Brussels time. Later origins are not fully observed on 2 November.

No scored truth was imputed, and every eligible context had zero missing or
imputed actual hours. Context timestamps are strictly before forecast delivery;
future actuals are never passed to the model. The actual context uses a common
archived/revised history, not reconstructed issue-time observation vintages.

The target remains the online DE_LU collection area. Capacity selection uses the
German registry; a separate Luxembourg point was not introduced.

The 2 November inventory contains 6,461,221 geolocated in-service/temporarily
stopped solar units totalling 117.092 GW net (132.278 GWp gross). The pre-period
filter retains 6,173,661 units and 111.803 GW net. Public postcode centroids cover
about 39% of net capacity whose exact locations are private. Unlocated units
account for at most 53.662 MW across all statuses, giving a conservative located
capacity lower bound of 99.95%. Selection uses capacity, not evaluation errors.

These are paired retrospective ablations: Open-Meteo historical forecasts stitch
short-lead forecasts and cannot reproduce the original day-ahead weather vintage
at all new locations. Absolute scores differ from the live leaderboard. Both
arms share this reconstruction; interpret the changes as conditional on it.
Seven-origin circular block-bootstrap intervals are exploratory 95% intervals,
without adjustment for comparisons across models. Monthly tables use forecast
origin month. Runtime summaries are local CPU medians; initial loads are retained
in the raw timing records and these timings are not cloud-cost estimates.
