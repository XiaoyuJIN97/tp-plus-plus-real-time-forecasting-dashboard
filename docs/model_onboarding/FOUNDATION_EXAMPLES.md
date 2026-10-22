# Worked examples: T0 Gamma and Timestral

Run commands from the repository root on the Linux CPU benchmark profile. These
examples reuse the **currently deployed checkpoint revisions and native calls**;
they do not assert that either legacy model has passed the new admission policy.
Example keys have an `_onboarding_example` suffix to avoid overwriting live rows.
Do not enable those duplicate keys beside the existing models in production.

Code:

- [Manifest/input preparation](../../scripts/onboarding_foundation_example.py)
- [Native adapters](../../scripts/onboarding_foundation_adapter.py)
- [Existing runtime being reproduced](../../src/rt_forecast_dashboard/models/foundation_runtime.py)
- [Existing archive replay preparation](../../scripts/replay_foundation_online.py)
- [Previous replay results and limitations](../foundation_online_since_2026-07-24/RESULTS.md)
- [Previous input hashes/source revision](../foundation_online_since_2026-07-24/input_manifest.json)
- [Previous replay configuration](../foundation_online_since_2026-07-24/replay_config.json)

The previous replay used reconstructed inputs, including weather supplements
without certified issue-time vintages. Preserve that label. Those files/results
illustrate lineage and comparison, but cannot automatically satisfy the new
strict input-availability gate; missing provenance needs the reviewed exception
process described in the main guide.

| Family | Checkpoint | Immutable weights revision | Native input/output |
| --- | --- | --- | --- |
| T0 Gamma | `theforecastingcompany/t0-gamma` | `e08aee083778f1f32da4974c0a34276776a35c52` | Context `(1,2208)`, covariates `(1,C,2232)`; median `(1,24)` |
| Timestral | `mistralai/Timestral-209M` | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` | Context `(1,1,2208)`, covariates `(1,C,2232)`; median `(1,1,24)` |

Both examples return nonnegative MW point forecasts. They use the same covariate
selection as the current online models, obtained from `covariates_for`; the
generated manifest saves the exact list. Start with BE/load, then explicitly
expand scope and repeat qualification for each additional zone/target.

## Data and outputs at every step

| Step | Required local data | Produced evidence |
| --- | --- | --- |
| Package/review | Exact checkpoint, vendor code, complete environment lock, license/data-rights records | `manifest.yml`, disabled `registry-entry.yml`, `requirements.lock`, preparation log, local checkpoint |
| Offline | 2024/2025 origin inventory; 2,208 past hours per origin (including 2023 context); 24 future covariate hours; separately held truth; matched reference predictions | Frozen `windows.jsonl`, `references.csv`; `forecasts.csv`, `runtime.csv`, `run.json`, worker log |
| Replay | Inventory covering the exact retained online origins, original input vintages and availability evidence; matching incumbent history | Same frozen/run files, tagged as replay |
| Ensemble | Offline candidate run plus component-level offline reference predictions and observed outcomes | With/without-candidate `ensemble.json` |
| Shadow | Fresh daily snapshots, once each real day for seven consecutive dates; no future truth required at inference | Seven daily run folders, combined `shadow` evidence |
| Admission | All previous evidence; completed human reviews | Generated `admission.json`, reviewed registration PR |
| Publication | Passing admission, complete admitted replay/daily batch, separately maintained actuals | Candidate rows, rebuilt ensemble/leaderboard/probabilistic artifacts |

No raw weather or power data is bundled into this repository: the examples read
operator-supplied, permitted local archives. They never invoke collectors. A
checkpoint is downloaded only by the explicit preparation command below.

## 0. Generate each candidate and install its environment

```sh
python -m venv .onboarding-host
.onboarding-host/bin/pip install -r requirements-onboarding.txt scipy
export PYTHONPATH=src:.
for family in t0_gamma timestral; do
  .onboarding-host/bin/python scripts/onboarding_foundation_example.py init \
    --family "$family" --zone BE --target load --output "staging/$family/package"
done
```

For each package, review the generated manifest and install dependencies in a
**clean, separate** candidate environment. This bootstrap reproduces the current
shared foundation runtime; it is not itself a complete lock:

```sh
for family in t0_gamma timestral; do
  python -m venv ".model-envs/$family"
  ".model-envs/$family/bin/pip" install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
  ".model-envs/$family/bin/pip" install -r scripts/requirements-foundation-online.txt pandas PyYAML
  ".model-envs/$family/bin/pip" check
  ".model-envs/$family/bin/pip" freeze --all > "staging/$family/package/requirements.lock"
done
```

Review and retain the exact generated transitive lock, Python version and wheel
index. Reinstall that lock into a second clean Linux environment before collecting
admission evidence. The gate verifies the actual installed versions and VCS
revisions. Do not use the dashboard environment or the toy example's lock.
Separate minimal vendor-specific locks may later replace this shared bootstrap.

License fields deliberately remain `unknown`, and operations/reviewer fields
remain unapproved. Check the pinned code and weights terms and all input-data
rights before research; complete the production review before admission.
Timestral's company/employer condition needs an explicit assessment. Merely
copying an existing model's manifest is not approval. Include the retrospective
2024/2025 pretraining-overlap limitation in the review.

After evaluation is permitted:

```sh
for family in t0_gamma timestral; do
  .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding prepare \
    --manifest "staging/$family/package/manifest.yml" \
    --python ".model-envs/$family/bin/python" \
    --output "staging/$family/preparation.json"
done
```

## 1. Prepare offline data and run 2024/2025

Prepare one CSV/Parquet context and future pair for **each forecast origin**.
Reuse those files for both models and all reference runs. The inventory is:

```csv
zone,target,run_date,origin_utc,inputs_available_at_utc,context_path,future_path
BE,load,2024-01-02,2024-01-02T00:00:00Z,2024-01-01T23:45:00Z,2024-01-02/context.parquet,2024-01-02/future.parquet
```

This is a schema example, not evidence that inputs were actually available at
23:45. Paths are relative to the inventory. `run_date` is the Brussels date of
the first delivery hour, not the download date. Use each real forecast's exact
origin; do not round or shift online runs to midnight.

Required frame contents:

```python
import pandas as pd
import yaml
from rt_forecast_dashboard.model_onboarding.contracts import frames

spec = yaml.safe_load(open('staging/t0_gamma/package/manifest.yml'))
covs = spec['scope'][0]['covariates']  # same list for Timestral in this scope
context = pd.read_parquet('archives/offline/2024-01-02/context.parquet')
future = pd.read_parquet('archives/offline/2024-01-02/future.parquet')
# context columns: timestamp, actual_mw, *covs; 2208 consecutive hourly UTC rows
# future columns: timestamp, *covs; 24 consecutive hourly UTC rows
# Optional future.actual_mw supplies scoring truth, never model input.
past, known_future = frames(context, future, covs)
assert len(past) == 2208 and len(known_future) == 24
assert 'actual_mw' not in known_future
```

For load, covariates are the selected four-point weather group and optionally
`tso_forecast_mw`. Solar uses radiation/temperature; wind uses speed and direction
sine/cosine, plus the configured TSO forecast. Inspect the generated list rather
than guessing names. Raw collector weather fields require the existing feature
engineering first. `scripts/replay_foundation_online.py` shows the current
archive joins/context repair, but its hard-coded 2026 paths are **not** a 2024/2025
loader. Supply historical archives for those years and retain repair records.
Do not substitute observations for known-future weather or the TSO forecast.

References must already have been run on these same prepared windows. Supply a
local CSV with one row per model/delivery hour:

```csv
zone,target,run_date,timestamp,model,forecast_mw,actual_mw
BE,load,2024-01-02,2024-01-02T00:00:00Z,chronos2_online,10000,10100
```

Provide **all 24 hours for every declared reference and origin**, not just this
illustrative row. The generated load manifest declares Chronos2, Ridge,
TimesFM3, existing T0 Gamma, existing Timestral and TSO. Using existing identical
T0/Timestral as references can prevent their duplicate examples passing the 2%
improvement gate; that is expected. This is a reproducible example, not a claim
that re-adding an incumbent supplies new predictive value. A genuinely new
candidate's references should be all relevant eligible incumbents, with any
exclusion explained and reviewed. Reusing TimesFM3 inputs/outputs also requires
the applicable rights review.

```sh
for family in t0_gamma timestral; do
  .onboarding-host/bin/python scripts/onboarding_foundation_example.py freeze \
    --manifest "staging/$family/package/manifest.yml" \
    --inventory archives/offline/inventory.csv \
    --references archives/offline/references.csv --output "staging/$family/offline-inputs"
  .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding run \
    --stage offline --manifest "staging/$family/package/manifest.yml" \
    --windows "staging/$family/offline-inputs/windows.jsonl" \
    --python ".model-envs/$family/bin/python" --output "staging/$family/offline"
done
```

Use full daily years where possible; the minimum is 60 runs/year, all 12 months,
five runs/month, per scope. Read `runtime.csv` including failed windows for
cold/warm runtime, RSS and success rates. The admission report computes annual
accuracy and resource gates. An unsuccessful real model is an evaluation result,
not a reason to weaken thresholds in the sample.

## 2. Replay the retained online history

Build `archives/replay/inventory.csv` from the **current retained artifact's exact
scope/origin set**, and `archives/replay/references.csv` from its component rows
joined with observed actuals where available. Keep unobserved latest forecasts.
The artifact supplies reference results, not the historical input vintages.
Recover those separately from the collector snapshots used by the original runs.

For example, inspect the local dashboard history before preparing the inventory:

```python
import pandas as pd
history = pd.read_parquet('data/dashboard/forecasts_recent.parquet')
print(history.loc[(history.zone == 'BE') & (history.target == 'load')]
      .groupby('model').run_date.agg(['min', 'max', 'nunique']))
# Once the local artifact has canonical raw model keys and observed actuals:
import yaml
from pathlib import Path
spec = yaml.safe_load(open('staging/t0_gamma/package/manifest.yml'))
refs = history.loc[(history.zone == 'BE') & (history.target == 'load') &
                   history.model.isin(spec['reference_models'])].copy()
Path('archives/replay').mkdir(parents=True, exist_ok=True)
refs[['zone', 'target', 'run_date', 'timestamp', 'model', 'forecast_mw', 'actual_mw']]\
    .to_csv('archives/replay/references.csv', index=False)
# Deduplicate conflicting revisions explicitly, not by silently selecting a row.
```

Repeat the `freeze` and `run` commands from step 1 with `archives/replay`,
`replay-inputs`, `--stage replay`, and output `staging/$family/replay`.
Never fabricate availability timestamps for missing vintages. Record unavailable
archives as a limitation/blocker; do not relabel reconstructed data as live.

## 3. Compare ensembles

```sh
for family in t0_gamma timestral; do
  .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding ensemble-check \
    --manifest "staging/$family/package/manifest.yml" --candidate "staging/$family/offline" \
    --baselines "staging/$family/offline-inputs/references.csv" \
    --output "staging/$family/ensemble.json"
done
```

Required data is the offline component predictions and truth from step 1; no new
collector requests are necessary. Read annual with/without-candidate MAE and
membership in the JSON. The existing earlier-only ensemble weighting is reused.
Do not register both the live model and its identical example as new members.

## 4. Seven real daily shadow runs

Each real day, prepare `archives/shadow/DATE/inventory.csv` with the **fresh shared
snapshot**, its original availability record, and all declared references in
`references.csv`. Future actuals may be empty. Use the following once per day;
`DATE` is supplied explicitly from that day's actual delivery run:

```sh
DATE=YYYY-MM-DD
for family in t0_gamma timestral; do
  .onboarding-host/bin/python scripts/onboarding_foundation_example.py freeze \
    --manifest "staging/$family/package/manifest.yml" \
    --inventory "archives/shadow/$DATE/inventory.csv" \
    --references "archives/shadow/$DATE/references.csv" \
    --output "staging/$family/shadow-inputs/$DATE"
  .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding run \
    --stage shadow --manifest "staging/$family/package/manifest.yml" \
    --windows "staging/$family/shadow-inputs/$DATE/windows.jsonl" \
    --python ".model-envs/$family/bin/python" --output "staging/$family/shadow-days/$DATE"
done
# After seven consecutive actual daily runs:
for family in t0_gamma timestral; do
  .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding combine-runs \
    --directories staging/"$family"/shadow-days/* --output "staging/$family/shadow"
done
```

Keep the core daily job independent. Test timeouts/crashes/missing inputs and
compare request counts/unchanged incumbent rows. Record the results and rollback
evidence in the manifest; synthetic or historical batches cannot replace shadow.

## 5. Admission and controlled publication

Complete the legal and operational reviews, then run for each family:

```sh
for family in t0_gamma timestral; do
  .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding gate \
    --manifest "staging/$family/package/manifest.yml" \
    --offline "staging/$family/offline" --baselines "staging/$family/offline-inputs/references.csv" \
    --replay "staging/$family/replay" --replay-baselines "staging/$family/replay-inputs/references.csv" \
    --shadow "staging/$family/shadow" --ensemble-report "staging/$family/ensemble.json" \
    --output "staging/$family/package/admission.json"
done
```

A report can correctly fail: these examples are not pre-approved. For a real
admission, choose final durable repo paths **before running evidence**, generate
the package there, and commit manifest, exact lock, adapter and passing report.
Review the disabled registration and independent daily job, then enable through
the admission PR. Absolute checkpoint paths must point to the prepared cache on
the intended runner. Changed config/paths invalidate inference fingerprints.

After admission and registration only, the explicit importer is:

```sh
family=t0_gamma  # or timestral, using its own approved manifest
.onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding publish \
  --manifest "staging/$family/package/manifest.yml" \
  --run-directory "staging/$family/replay" --data-dir data
```

Use an admitted fresh daily `shadow` result for daily publication. Importing is
not deployment: rebuild ensembles, dashboard/probabilistic artifacts and commit
through the existing publisher as described in the [main guide](README.md#5-admission-pr-and-publication).
Verify live artifact membership, latest run and panels. Stop the candidate job
and disable its registration to roll back, then rebuild affected artifacts.

## No-download contract check

```sh
PYTHONPATH=src:. .onboarding-host/bin/python -m pytest -q tests/test_onboarding_examples.py
```

These tests generate full-size synthetic input frames in temporary directories,
check both native tensor layouts with stubbed vendor calls, verify frozen truth
separation and missing-reference rejection, and check checkpoint parity with the
current runtime. They do **not** measure real model accuracy, memory, runtime,
download/load compatibility or seven-day operation. Real archives and native
environment runs above remain necessary.
