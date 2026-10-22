# Adding a forecasting model

For step-by-step code and required datasets using the current T0 Gamma and
Timestral checkpoints, see [the worked examples](FOUNDATION_EXAMPLES.md).

This is the standard process for a new **local CPU TSFM**. It provides reusable
input/export, isolated inference, metrics/admission, ensemble and publication
commands. The operator still writes the vendor-specific adapter and checks
licenses/data provenance. API-hosted or GPU models need a reviewed profile/code
extension; this process does not silently authorize extra requests or GPU costs.

Existing models have not been retrospectively certified by this process. In
particular, existing TimesFM3 licensing concerns are not resolved by these tools.
The GitHub guard grandfathers unchanged legacy registrations technically, not
legally. This change enables no new production model and runs no paid API.

## Stages and owners

| Stage | Required evidence | Exit condition |
| --- | --- | --- |
| 0. License and feasibility | Exact weights/code licenses, data rights, entity/revenue conditions, immutable checkpoint, pinned environment, covariate support | Evaluation allowed; commercial/production review must pass before admission |
| 1. Offline 2024 and 2025 | Shared frozen windows and reference forecasts, annual/macro MAE, cold/warm timings, peak RAM, environment record | Performance, completeness, reliability and CPU gates pass |
| 2. Historical mock replay | Exact retained online delivery windows, original input availability/provenance, failures and scope coverage | Full alignment; no invented/backfilled model predictions |
| 3. Ensemble qualification | With/without candidate on the same historical windows, earlier-only fitted weights | Candidate is included and ensemble MAE does not regress beyond policy |
| 4. Shadow run | At least seven consecutive successful real daily runs per approved zone/target; request/failure/rollback checks | No production publication during shadow; operations review recorded |
| 5. Admission and rollout | Generated report + reviewer evidence in PR, isolated job wiring, staged import and UI checks | CI passes, reviewed PR merges, explicit publication and artifact rebuild |
| 6. Monitoring/rollback | Daily completeness, latency/RAM, observed accuracy, coverage and provenance | Disable/remove candidate and rebuild affected ensembles if it regresses |

The model owner supplies datasets, adapter and evidence. A second operator
reviews the license conditions, reference selection, input availability, hardware
measurements, ensemble result, request counts and rollback before merging.
The report is evidence/checking infrastructure, not a cryptographic attestation
or a substitute for PR review. Require the `Model onboarding contract and admission` check and operator review
for model-code/config PRs where repository rules permit. Preserve the authorized
automated data-publisher path: a blanket require-PR rule can break daily data
commits. This task does not change repository protection/bypass settings.

## Initial admission defaults

All values are in [config/model_onboarding.yml](../../config/model_onboarding.yml).
These are **starting operational choices**, not universal scientific thresholds.
Change them only in a reviewed PR; changing policy invalidates admission reports.

| Criterion | Default |
| --- | --- |
| Forecast contract | 2,208 hourly context values; 24 hourly predictions in MW; UTC grids |
| Historical coverage | Both 2024 and 2025; >=60 matched runs/year/scope, all 12 months, >=5 runs/month; full daily years preferred |
| Fair comparison | Identical origins, horizons, outcomes and input settings; >=95% of planned offline runs matched |
| Annual non-regression | Candidate MAE <=1.05 × best declared reference MAE in **each year** |
| Added predictive value | Mean of the two annual relative MAE improvements >=2%, **per approved zone/target** |
| Ensemble | With-candidate MAE <=1.05 × without-candidate MAE in each evaluated scope/year |
| Reliability | >=99% successful windows overall **and for every scope**, per stage |
| Replay | Exact complete baseline window set for each scope, including unobserved latest forecasts |
| Runtime | Warm p95 <=30 seconds/window; cold initialization + first prediction <=180 seconds |
| Daily budget | Sum of scoped warm p95 values + worst cold time <=600 seconds/model |
| Memory | <=4,096 MiB process-family RSS, plus worker high-water RSS check |
| Benchmark profile | Linux, two CPU math threads; separate candidate interpreter; cached checkpoint |
| Shadow | >=7 consecutive successful real dates/scope; run recorded within 12 hours after origin |

The quality threshold uses the best **declared** reference on common rows,
not an unverified claim of beating every model. Include all approved relevant
incumbents in `reference_models`; explain omissions in review. Report RMSE/bias,
extreme-weather and seasonal behavior alongside MAE if important for the use
case. The standard automatic gate intentionally has no hidden manual pass flag.
A slower or weaker standalone model that improves an ensemble needs a separately
reviewed policy exception, not an edited `passed` field.

Evaluate 2024/2025 by forecast-origin year, including necessary 2023 context and
warmup forecasts. Report model-release/pretraining overlap: a 2026 foundation
model can have seen 2024/2025 data. Retrospective scores are not necessarily
out-of-training evidence. The later live shadow period is therefore important.

## 1. Candidate package and manifest

Copy [manifest.yml](../../examples/model_onboarding/manifest.yml),
[registry-entry.yml](../../examples/model_onboarding/registry-entry.yml) and the
[runnable protocol example](../../examples/model_onboarding/demo_adapter.py).
The example is a seasonal-naive toy, **not a new foundation model**. Its blank
license review and placeholder revision deliberately fail production admission.

Use `docs/models/<model_key>/manifest.yml` for the reviewed manifest. Implement:

```python
def prepare_model(config):
    # Download only from the publisher, only when permitted, at config['revision'].
    # Pre-cache the weights; do not put model weights into GitHub artifacts.
    ...

def create_adapter(config):
    # Load the exact checkpoint locally; no network calls during inference.
    return Adapter(config)

class Adapter:
    def predict(self, features, target, context=None, covariates=None):
        # context: timestamp, actual_mw and selected past covariates.
        # features: timestamp and selected known-future covariates; NO actuals.
        # Return shape (24,), finite/nonnegative MW, matching input order.
        ...
```

Do vendor tensor layout, scaling and quantile/point extraction here. Never
silently replace a failed model with TSO/persistence under the candidate's name.
Reject unsupported frequency, shapes or covariates explicitly. Keep mapping in
the reviewed `adapter_source` file (other local helper changes also need review).

Install each model's **complete exact lock** into its own environment, never into
the dashboard/core environment. Lock transitive dependencies and VCS commits;
record Python/platform and `pip check`. Admission compares installed versions/VCS
revisions against the complete lock (excluding optional venv bootstrap tools),
not just the lock filename/hash. Use platform-specific `pip freeze --all` style
locks; unpinned requirements or environment-marker branches are not accepted. The sample lock is for the demo only.
Production CPU wheel/index selection must be reproducible and documented.

```sh
python -m venv .onboarding-host
.onboarding-host/bin/pip install -r requirements-onboarding.txt scipy
python -m venv .model-envs/MODEL
.model-envs/MODEL/bin/pip install -r docs/models/MODEL/requirements.lock
.model-envs/MODEL/bin/pip check
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding prepare \
  --manifest docs/models/MODEL/manifest.yml --python .model-envs/MODEL/bin/python \
  --output staging/MODEL/preparation.json
```

Use `PYTHONPATH=src` for commands when not installing the project. Preparation
permits publisher downloads in a separate bounded step; it receives no upstream
API credentials. The inference worker blocks Python socket connections and sets
Hugging Face/Transformers offline flags. This is defense against accidental
requests, **not a security sandbox for malicious native code**. Run only reviewed
adapters. RSS monitoring requires process-inspection permission (Linux CI works;
restricted macOS environments may need permission). CPU workers are POSIX-only.

## 2. Shared, frozen input windows

Prepare inputs **once per window**, then reuse the same snapshot for every
candidate/reference. Do not fetch Open-Meteo/ENTSO-E once per model. Pin source
repository revisions, dataset hashes, cutoff convention and covariate policy.

Export a plan CSV with these fields:

```text
zone,target,run_date,origin_utc,inputs_available_at_utc,context_path,future_path,covariates
```

Paths are relative to the plan file. Context/future files are Parquet or CSV.
Context has `timestamp`, `actual_mw`, and exactly the covariates declared in the
manifest's zone/target scope. Future has `timestamp` and those known covariates.
Use the existing `context_quality.repair_context` guard during preparation and
retain its imputation records; do not silently fill long missing periods.
For historical scoring the future file can additionally have `actual_mw`; the exporter keeps
truth separate and the worker never receives it. `covariates` in the plan is a
quoted JSON array, including `[]` for a genuinely univariate model.

```sh
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding export-windows \
  --plan staging/MODEL/offline-plan.csv --output staging/MODEL/offline-windows.jsonl
```

Each JSONL window records zone/target/run_date, UTC origin and actual input
availability, context/future records, covariate names, separate actuals and source
hashes. Context ends exactly one hour before the first delivery timestamp.
Do not claim reconstructed weather/actual revisions were available earlier than
proven. Snapshot creation time or weather-model initialization time alone does
not establish the original publication time. If vintages are unavailable,
document retrospective limitations and seek a reviewed exception; do not invent
`inputs_available_at_utc` to pass the gate.

Reference CSV schema:

```text
zone,target,run_date,timestamp,model,forecast_mw,actual_mw
```

Every reference must use the same windows/settings; the evaluator rejects
conflicting actuals and duplicate keys. It reports annual candidate MAE against
the lowest reference MAE on the same complete windows. Replay reference CSVs
must contain the entire current retained online history for the approved scope;
derive its start/end from the dashboard artifact rather than a stale hard-coded
date. Preserve actual outcomes separately from inference inputs.

## 3. Offline and replay execution

```sh
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding run \
  --stage offline --manifest docs/models/MODEL/manifest.yml \
  --windows staging/MODEL/offline-windows.jsonl --python .model-envs/MODEL/bin/python \
  --output staging/MODEL/offline
# Repeat with --stage replay and the aligned online windows; use a new output dir.
```

Outputs are `forecasts.csv`, all attempted windows in `runtime.csv`, `worker.log`
and `run.json` (input/model/lock/source/artifact fingerprints, environment and
completion status). Failed windows remain explicit; no fake predictions are
written. Startup/download/install time is distinct from warm inference; cold
first-prediction cost is still included in the cold/daily budget. Keep failed
attempts when calculating success and timings, not just successful runs.

For GitHub, use [Evaluate a candidate model](../../.github/workflows/evaluate-candidate-model.yml).
Prepare a same-repository `frozen-windows` artifact containing `windows.jsonl`
(no model weights), then dispatch with its run ID, manifest and stage. Select the candidate Python version independently of the host. The job
installs only the candidate's lock in its own venv, downloads permitted weights
from the publisher, evaluates with networking blocked in the worker, and retains
staging evidence. It never commits or publishes candidate forecasts. Confirm
rights to retain input/output artifacts; use a private project or local runner
when public artifact retention is not allowed. Artifact lifetime is 90 days;
archive permitted evidence durably and put its location in the review record.

Large studies can be sharded into <=90-minute jobs. Use the same manifest,
policy and lock, then combine **non-overlapping** parts:

```sh
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding combine-runs \
  --directories staging/MODEL/offline-part-1 staging/MODEL/offline-part-2 \
  --output staging/MODEL/offline
```

## 4. Ensemble and shadow qualification

```sh
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding ensemble-check \
  --manifest docs/models/MODEL/manifest.yml --candidate staging/MODEL/offline \
  --baselines staging/MODEL/offline-references.csv --output staging/MODEL/ensemble.json
```

This runs the existing mean/median/learned ensemble policy with and without the
candidate in an in-memory preview store, including automatic component discovery
and earlier-only learned weights. It writes no production forecasts. The report
contains matched annual MAEs and fingerprints; insufficient observed scope/year
coverage or excessive ensemble regression fails. Provide all reference components
needed by the configured ensemble; do not pass an ensemble-only baseline CSV.
Also review leave-one-out contribution, changed weights and warmup behavior.

Run `--stage shadow` on freshly prepared windows **each real day**, keeping
outputs separate. Combine those daily folders after seven consecutive successes.
The gate checks recorded timing as well as dates; replaying seven old dates in
one batch is not a live shadow trial. Review accuracy when actuals are observed.
Test missing inputs, timeout, OOM, crash, malformed outputs and dependency failure;
verify incumbents' forecasts and collector request counts remain unchanged.
Workers receive no ENTSO-E/Open-Meteo credentials and cannot fetch missing data.

Document the test evidence and rollback in `operations` and the legal/pretraining
review in `review`. Updating these reviews does not force repeating unchanged
inference: run identity excludes review/license/operations fields. The final
admission report fingerprints the complete reviewed manifest.

## 5. Admission, PR and publication

```sh
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding gate \
  --manifest docs/models/MODEL/manifest.yml \
  --offline staging/MODEL/offline --baselines staging/MODEL/offline-references.csv \
  --replay staging/MODEL/replay --replay-baselines staging/MODEL/online-references.csv \
  --shadow staging/MODEL/shadow --ensemble-report staging/MODEL/ensemble.json \
  --output docs/models/MODEL/admission.json
```

Exit 0 means all gates passed; otherwise read `failed_checks` and fix/restrict the
scope or reject the model. Never hand-edit a report to pass it. Commit manifest,
lock, adapter and compact report in a PR. Add the enabled isolated registration
only after the evidence passes. New/re-enabled/revised registrations are checked
against the PR base. Source, environment, policy or scope changes invalidate the
relevant evidence. Operator review is still required to assess references,
artifact authenticity, license conditions and lineage.

The approved registration declares the exact scope/covariates, interpreter and
resource bounds. Scoped models are honored by daily inference, completeness,
ensemble discovery and unknown-family dashboard display. Keep
`required_for_completeness: false` for a newly added optional job: the core daily
job must still publish if that model fails. The candidate's own job explicitly
checks `check_forecast_complete --models MODEL`; that check includes optional
models. Registry/ensemble discovery **does not schedule model inference**.

Wire a separate bounded candidate job using shared prepared windows. Use the
frozen-window `run` command above, then explicitly import only its successful
admitted replay/shadow results. Do not rerun live collectors or use API fallbacks
to compensate for missing candidate inputs:

```sh
PYTHONPATH=src .onboarding-host/bin/python -m rt_forecast_dashboard.model_onboarding publish \
  --manifest docs/models/MODEL/manifest.yml --run-directory staging/MODEL/replay \
  --data-dir data
# Use a fresh --stage shadow result for an admitted daily run.
```

Offline-test files and failed/partial candidate batches cannot be published.
Replay is tagged `historical_mock_replay`; successful admitted shadow/daily
results are tagged `online`. The importer replaces only that model's groups,
never incumbent groups, and revalidates manifest/report/run fingerprints.
It requires a complete batch; fix/re-run failures or restrict scope explicitly.
It does not commit to GitHub or start a deployment by itself.

After import, use existing history repair and artifact builders with the observed
actuals file, then publish through `scripts/commit_dashboard_data.sh`:

```sh
PYTHONPATH=src python -m rt_forecast_dashboard.pipeline.build_selected_ensembles \
  --end YYYY-MM-DD --recent-days 1 --repair-history --actuals PATH_TO_ACTUALS.parquet
PYTHONPATH=src python -m rt_forecast_dashboard.pipeline.build_dashboard_artifacts \
  --retention-days 90 --actuals PATH_TO_ACTUALS.parquet
```

New enabled eligible models join the ensemble automatically; TSO policy remains
separate. History repair invalidates changed component signatures. Check the
published artifact's latest model groups and ensemble membership, shared-run
leaderboard and probabilistic panels before announcing rollout. A passing local
test or merged code alone is not proof the live artifact has updated.

## Rollback and operational checks

- Disable the candidate in the registry and stop its independent job. Remove or
  quarantine its public candidate rows if required; preserve permitted research
  evidence separately. Other model groups must remain unchanged.
- Rebuild affected ensembles from retained eligible members, then leaderboard and
  probabilistic artifacts. Changing component signatures triggers history repair.
- Confirm a model failure cannot terminate the core job, mutate shared inputs,
  add unreviewed upstream requests, or overwrite another model's groups.
- Re-review after changing weights/revision, runtime, covariates/scope, terms or
  commercial context (including employer revenue conditions). Watch latency,
  failure rate, observed errors and interval coverage after release. For native quantiles, additionally check ordering
  and calibration before exposing them; the existing conformal bands remain the
  default probabilistic presentation.

Use [PR_CHECKLIST.md](PR_CHECKLIST.md) when opening the admission PR.
