# Weather-vintage audit and retrospective forecast repair — 10 November 2026

Historical replay formerly joined merged annual weather by delivery time. That
cannot demonstrate availability at a simulated cutoff. The new selector uses
original collector snapshots whose manifest AND per-row collection timestamps
are no later than the 18:00 Europe/Brussels cutoff. Future hours come from one
complete eligible collection. Historical-only backfills cannot provide future
weather. Past weather may combine eligible collections, choosing the latest
complete row at each timestamp. Missing evidence/coverage fails closed.

Collection establishes that the provider had made those values accessible by
that time; it does not establish the NWP initialization time. The repair certifies
weather availability only. Historical TSO and actual revision vintages remain
retrospective and are not newly certified. Model settings and the 2,208-hour
context / next-day-midnight horizon are unchanged.

## Frozen audit

- `source_manifest.json`: original dashboard and weather Git revisions, manifest
  hash, checkpoint revisions and frozen non-weather input sources.
- `mock_weather_audit.csv`: every stored retrospective model group, including
  legacy groups outside the current dashboard window.
- `current_dashboard_audit.csv`: the subset present in the frozen 90-day artifact.
- `weather_provenance.json`: eligible snapshot paths, hashes and collection times.
- `before_*`: original score snapshots for comparing the repair.

`status` describes archive availability. `weather_input_required=false` means the
model does not use weather (e.g. persistence, TSO-only); unavailable weather does
not invalidate such a model. For an ensemble this flag is conservatively true.

The archive starts in August for BE/FR/DE and October for newer areas. Later
historical backfills cannot establish earlier availability. Unsupported originals
remain preserved; they must not be relabelled as verified or silently replaced
with values from the annual tables. Genuinely online predictions are preserved.

## Reproduction

Check out the weather revision recorded in `source_manifest.json`, including
`data/update_manifest.csv` and `data/updates`, and supply the frozen non-weather
inputs recorded there. Use isolated staging directories:

```sh
PYTHONPATH=src:. python scripts/audit_mock_weather_vintages.py \
  --data data --weather /path/to/weather-snapshot-checkout --output staging/audit
FORECAST_WEATHER_SNAPSHOT_ROOT=/path/to/weather-snapshot-checkout \
  PYTHONPATH=src:. python scripts/batch_replay_midnight.py \
  --source data --inputs /path/to/frozen-inputs --output staging/replay \
  --plan staging/eligible-native-plan.csv --start 2026-08-08 --end 2026-10-06
PYTHONPATH=src:. python scripts/stage_weather_vintage_repair.py \
  --source data --replay staging/replay --audit staging/audit/mock_weather_audit.csv \
  --output staging/publication
```

Native inference needs the existing pinned model environments/checkpoints. The
eligible plan must contain only original retrospective groups with available
weather, excluding weather-independent baselines and generated ensembles.
Recompute affected historical ensembles separately; filter publication to the
original historical ensemble keys so no genuinely online ensemble is replaced.
Invalidate the daily score cache, then rebuild point and conformal artifacts and
update the hosted data revision before publishing.

## Treatment of unavailable snapshots

The operator explicitly chose to keep unsupported mock runs in dashboard
comparisons and keep labels internal. No UI labels, warnings, or filtering are
added. `weather_archive_audit` and `weather_vintage_status` are internal fields;
these distinguish corrected weather from unavailable evidence and models that
use no weather. Thus the published leaderboard remains a mixture of corrected
and unverified retrospective history; it is not a wholly certified backtest.

```sh
PYTHONPATH=src:. python scripts/rebuild_repaired_mock_ensembles.py \
  --staged staging/publication --scratch staging/ensemble-work --first 2026-08-08
```


## Published repair counts

- 2,755 native model groups regenerated successfully; 2,689 changed numerically.
- 2,195 historical ensemble groups rebuilt; 1,029 changed numerically.
- Across both steps, 108,248 prediction hours changed in 3,718 model groups.
- All 708,032 stored forecast keys and their scoring actuals were preserved.
- All 71,016 online-labelled prediction rows were preserved within serialization
  tolerance (absolute 1e-8 MW, relative 1e-12).
- Both point and probability leaderboards retain 216 score rows; unsupported
  historical runs remain in comparisons per operator preference.
- 51 focused regression tests passed; isolated DST worker passed with required
  macOS process-inspection access; contract/ensemble/native CPU CI also checked.

`online_label_timing_audit.csv` separately records later-day generations that
were already labelled online. Some are delayed daily jobs; the label alone does
not prove a forecast was generated before every delivery hour. These originals
were not overwritten by this weather repair. Past-date catch-ups now have an
explicit retrospective label and strict weather selection.

See `prediction_comparison.csv`, `replacement_validation.json`,
`ensemble_validation.json` and `final_validation.json` for numerical checks.
Original native predictions and before/after score artifacts are retained here.
