# Model admission PR checklist

- [ ] License review covers exact code/weights, commercial/production use,
      distribution/outputs, attribution, entity conditions and input/API rights.
- [ ] Immutable model revision, complete exact dependency lock, separate venv,
      preparation/cache instructions and adapter source are committed.
- [ ] Scope and covariates are explicit. Inputs use UTC, 2,208 past hours and
      24 delivery hours; no future actuals, false availability or silent fallback.
- [ ] 2024/2025 evidence covers both full years or the policy's monthly sample;
      shared baselines/settings/outcomes, counts, annual MAE and timings are shown.
- [ ] Cold/warm CPU timings, peak RAM, platform, threads and failures are recorded.
- [ ] Mock history matches every retained online window in the approved scope;
      missing/reconstructed vintages and pretraining overlap are disclosed.
- [ ] With/without candidate ensemble report passes; prior-only fitting,
      six-hour weights, warmup behavior and missing components were reviewed.
- [ ] Seven consecutive real shadow dates per scope pass. Failure injection,
      unchanged incumbent outputs, zero model-driven collector calls and rollback
      evidence are linked in the manifest.
- [ ] Admission report was generated from these exact artifacts and reviewed;
      no manual passing flag or unreviewed threshold exception.
- [ ] Optional isolated daily job is wired with limits, cached weights and shared
      frozen inputs; registry inclusion alone does not schedule model inference.
- [ ] Replay/publication preserves provenance and incumbent groups; final
      completeness checks, ensemble membership and live artifacts are verified.
- [ ] Notices, operator owner, evidence archive and rollback procedure are ready.

Review failures are reasons to fix/restrict/reject, not to silently drop bad
windows from a report. Existing legacy registrations are not certified by this PR.
