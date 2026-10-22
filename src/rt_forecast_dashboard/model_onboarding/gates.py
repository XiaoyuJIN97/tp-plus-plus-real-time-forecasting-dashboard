"""Fail-closed admission from matched outcomes, resource evidence and reviews."""
from __future__ import annotations

from rt_forecast_dashboard.time_utils import forecast_horizon_hours
from datetime import UTC, datetime
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .contracts import digest, KEYS, inference_digest, staging_path
from .process import ROOT
from .environment import locked_environment


def _read_forecasts(path):
    path=Path(path)
    if not path.is_file():
        raise ValueError('Use a local frozen forecast CSV, not a URL')
    f = pd.read_csv(path)
    required = {*KEYS, 'model', 'forecast_mw', 'actual_mw'}
    if required-set(f):
        raise ValueError(f'Missing forecast columns: {sorted(required-set(f))}')
    f['timestamp'] = pd.to_datetime(f.timestamp, utc=True)
    f['run_date'] = f.run_date.astype(str)
    if f[[*KEYS,'model']].isna().any().any():
        raise ValueError('Missing forecast identity/timestamps')
    f['forecast_mw']=pd.to_numeric(f.forecast_mw,errors='raise')
    f['actual_mw']=pd.to_numeric(f.actual_mw,errors='raise')
    if not np.isfinite(f.forecast_mw).all() or f.forecast_mw.lt(0).any():
        raise ValueError('Invalid forecast values in comparison evidence')
    if np.isinf(f.actual_mw).any():
        raise ValueError('Infinite actuals are not valid scoring evidence')
    if f.duplicated([*KEYS, 'model']).any():
        raise ValueError('Duplicate forecast keys')
    return f


def evaluate(manifest_path, policy_path, *, offline, baselines, replay, replay_baselines, shadow, ensemble_report, output):
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    policy = yaml.safe_load(Path(policy_path).read_text())
    checks, metrics = [], []
    def check(name, passed, detail=''):
        checks.append(dict(name=name, passed=bool(passed), detail=str(detail)))
    scope = {(s['zone'], s['target']) for s in manifest['scope']}
    references = manifest['reference_models']
    check('valid_scope', bool(scope) and len(scope)==len(manifest['scope'])
          and all(isinstance(s.get('covariates'),list) for s in manifest['scope']))
    check('cpu_profile', manifest.get('adapter_config',{}).get('device')=='cpu')
    check('distinct_reference_models', len(set(references)) >= 2 and manifest['model_key'] not in references)
    check('immutable_revision', bool(re.fullmatch(r'(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})', manifest.get('revision',''))))
    check('pinned_adapter_config', manifest.get('adapter_config',{}).get('revision')==manifest.get('revision')
          and manifest.get('adapter_config',{}).get('model_id')==manifest.get('checkpoint'))
    lock = [line.strip() for line in (ROOT/manifest['requirements_lock']).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    check('pinned_dependencies', bool(lock) and all(re.fullmatch(r'[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^ ;]+(?:;.+)?',line) or re.search(r'git\+[^ ]+@[a-fA-F0-9]{40}(?:$|#)',line) for line in lock))
    for kind in ('code','weights','input_data'):
        license_ = manifest.get('licenses', {}).get(kind, {})
        check('license_'+kind, license_.get('commercial_use') in {'allowed','conditional'}
              and license_.get('conditions_met') is True and bool(license_.get('url')))
    review = manifest.get('review', {})
    check('human_review', all(review.get(key) for key in ('reviewer','evidence_url','training_data_overlap_assessment')))
    operations = manifest.get('operations', {})
    for key in ('ensemble_check_passed','failure_isolation_check_passed','rollback_tested'):
        check(key, operations.get(key) is True and bool(operations.get('evidence_url')))
    ensemble = json.loads(Path(ensemble_report).read_text())
    check('ensemble_performance', ensemble.get('passed') is True
          and ensemble.get('inference_sha256')==inference_digest(manifest_path)
          and ensemble.get('candidate_forecasts_sha256')==digest(Path(offline)/'forecasts.csv')
          and ensemble.get('baselines_sha256')==digest(baselines)
          and ensemble.get('policy_sha256')==digest(policy_path))
    all_runtime = []
    for stage, directory in [('offline',offline), ('replay',replay), ('shadow',shadow)]:
        directory = Path(directory)
        meta = json.loads((directory/'run.json').read_text())
        expected = dict(inference_sha256=inference_digest(manifest_path), policy_sha256=digest(policy_path),
                        requirements_sha256=digest(ROOT/manifest['requirements_lock']),
                        adapter_source_sha256=digest(ROOT/manifest['adapter_source']),
                        forecasts_sha256=digest(directory/'forecasts.csv'), runtime_sha256=digest(directory/'runtime.csv'))
        check(stage+'_completed',meta.get('completed') is True)
        check(stage+'_fingerprints', all(meta.get(k)==v for k,v in expected.items())
              and meta.get('stage')==stage and meta.get('model_key') == manifest['model_key'] and meta.get('revision') == manifest['revision'])
        check(stage+'_environment_lock',locked_environment(lock,meta.get('worker',{})))
        check(stage+'_hardware', policy.get('required_system') in {None,'any',meta.get('worker',{}).get('system')}
              and meta.get('threads')==policy['threads'])
        r = pd.read_csv(directory/'runtime.csv')
        if r.duplicated(['zone','target','run_date']).any():
            raise ValueError('Duplicate runtime windows')
        r['run_date'] = r.run_date.astype(str)
        success = r.success.astype(str).str.lower().eq('true')
        check(stage+'_reliability', len(r)>0 and success.mean() >= policy['minimum_success_fraction'], f'{success.sum()}/{len(r)}')
        all_runtime.append(r)
        f = _read_forecasts(directory/'forecasts.csv')
        check(stage+'_model', f.model.eq(manifest['model_key']).all())
        valid = np.isfinite(f.forecast_mw) & f.forecast_mw.ge(0)
        check(stage+'_finite_predictions', valid.all())
        complete = f.groupby(['zone','target','run_date']).agg(hours=('timestamp','nunique'))
        check(stage+'_complete_horizons', len(complete)>0 and all(row.hours == forecast_horizon_hours(key[2], policy['horizon_hours']) for key,row in complete.iterrows()))
        for zone,target in sorted(scope):
            planned = r[r.zone.eq(zone)&r.target.eq(target)]
            check(f'{stage}_{zone}_{target}_present', len(planned)>0)
            success_scope=planned.success.astype(str).str.lower().eq('true')
            check(f'{stage}_{zone}_{target}_reliability',len(planned)>0 and success_scope.mean()>=policy['minimum_success_fraction'])
            if stage == 'shadow':
                good = planned[planned.success.astype(str).str.lower().eq('true')]
                # Consecutive successful dates; a failed or missing scoped day breaks the streak.
                dates = pd.DatetimeIndex(pd.to_datetime(good.run_date.unique())).sort_values()
                streak = best = 0
                previous = None
                for date in dates:
                    streak = streak+1 if previous is not None and date-previous==pd.Timedelta(days=1) else 1
                    best = max(best, streak); previous = date
                check(f'shadow_{zone}_{target}_days', best >= policy['minimum_shadow_days'], best)
        if stage == 'shadow':
            observed = pd.to_datetime(r.observed_at_utc, utc=True)
            origins = pd.to_datetime(r.origin_utc, utc=True)
            check('shadow_live_timing', (observed>=origins).all() and (observed<origins+pd.Timedelta(hours=12)).all())
            continue
        b = _read_forecasts(baselines if stage=='offline' else replay_baselines)
        b = b[b.model.isin(references)]
        panel = b.pivot(index=KEYS, columns='model', values='forecast_mw')
        if set(references)-set(panel):
            raise ValueError(f'Missing reference models in {stage}')
        truth = b.groupby(KEYS).actual_mw.agg(['min','max'])
        check(stage+'_consistent_reference_actuals', np.allclose(truth['min'], truth['max'], equal_nan=True))
        panel = panel[references].join(truth['min'].rename('reference_actual')).reset_index()
        expected_keys = panel.dropna(subset=references).groupby(['zone','target','run_date']).filter(lambda p:len(p)==forecast_horizon_hours(p.run_date.iloc[0], policy['horizon_hours']))
        merged = f.merge(expected_keys, on=KEYS, how='inner', validate='one_to_one').dropna(subset=['actual_mw','reference_actual',*references])
        check(stage+'_identical_actuals', np.allclose(merged.actual_mw, merged.reference_actual, rtol=1e-6, atol=1e-6))
        # Compare every model on exactly the same complete delivery windows.
        merged = merged.groupby(['zone','target','run_date']).filter(lambda p:len(p)==forecast_horizon_hours(p.run_date.iloc[0], policy['horizon_hours']))
        for zone,target in sorted(scope):
            planned = r[r.zone.eq(zone)&r.target.eq(target)]
            candidate = f[f.zone.eq(zone)&f.target.eq(target)]
            common = merged[merged.zone.eq(zone)&merged.target.eq(target)]
            expected_scope = expected_keys[expected_keys.zone.eq(zone)&expected_keys.target.eq(target)]
            if stage == 'replay':
                # Baseline file defines the entire retained online history for requested scope.
                expected_set = set(map(tuple, expected_scope[KEYS].to_numpy()))
                actual_set = set(map(tuple, candidate[KEYS].to_numpy()))
                check(f'replay_{zone}_{target}_alignment', bool(expected_set) and actual_set==expected_set,
                      f'{len(actual_set)} candidate / {len(expected_set)} expected hours')
                continue
            gains = []
            for year in policy['offline_years']:
                p = planned[pd.to_datetime(planned.run_date).dt.year.eq(year)]
                c = common[pd.to_datetime(common.run_date).dt.year.eq(year)]
                n = c.run_date.nunique()
                check(f'offline_{zone}_{target}_{year}_runs', n >= policy['minimum_runs_per_year'], n)
                check(f'offline_{zone}_{target}_{year}_months', pd.to_datetime(c.run_date).dt.month.nunique() >= policy['minimum_months_per_year'])
                monthly=c.groupby(pd.to_datetime(c.run_date).dt.month).run_date.nunique()
                check(f'offline_{zone}_{target}_{year}_monthly_runs',not monthly.empty and monthly.min()>=policy['minimum_runs_per_month'])
                check(f'offline_{zone}_{target}_{year}_matched_fraction', len(p)>0 and n/len(p)>=policy['minimum_matched_fraction'])
                if c.empty:
                    gains.append(float('-inf')); continue
                candidate_mae = float((c.forecast_mw-c.actual_mw).abs().mean())
                reference_maes = {key:float((c[key]-c.actual_mw).abs().mean()) for key in references}
                best_key = min(reference_maes, key=reference_maes.get)
                best_mae = reference_maes[best_key]
                ratio = candidate_mae/best_mae if best_mae>0 else (1. if candidate_mae==0 else float('inf'))
                gain = 1-ratio
                gains.append(gain)
                metrics.append(dict(zone=zone,target=target,year=year,runs=n,mae_mw=candidate_mae,
                                    best_reference=best_key,best_reference_mae_mw=best_mae,mae_ratio=ratio))
                check(f'offline_{zone}_{target}_{year}_nonregression', ratio<=policy['maximum_year_mae_ratio'], ratio)
            check(f'offline_{zone}_{target}_improvement', np.mean(gains)>=policy['minimum_mean_mae_improvement'], np.mean(gains))
    runtime = pd.concat(all_runtime, ignore_index=True)
    elapsed = pd.to_numeric(runtime.elapsed_seconds, errors='coerce')
    peak = pd.to_numeric(runtime.peak_rss_mb, errors='coerce')
    check('valid_resource_measurements', np.isfinite(elapsed).all() and elapsed.ge(0).all() and np.isfinite(peak).all() and peak.gt(0).all())
    cold = runtime.cold_start.astype(str).str.lower().eq('true')
    cold_max = float(elapsed[cold].max()) if cold.any() else float('inf')
    check('cold_start_budget', cold_max<=policy['maximum_cold_seconds'], cold_max)
    check('memory_budget', peak.max()<=policy['maximum_peak_rss_mb'], peak.max())
    p95s = []
    for zone,target in sorted(scope):
        warm = elapsed[~cold & runtime.zone.eq(zone) & runtime.target.eq(target)]
        p95 = float(warm.quantile(.95)) if len(warm) else float('inf')
        p95s.append(p95)
        check(f'warm_{zone}_{target}_samples', len(warm)>=policy['minimum_warm_samples'], len(warm))
        check(f'warm_{zone}_{target}_p95', p95<=policy['maximum_warm_p95_seconds'], p95)
    daily_seconds = sum(p95s)+cold_max
    check('daily_budget', daily_seconds<=policy['maximum_daily_seconds'], daily_seconds)
    report = dict(model_key=manifest['model_key'], generated_at_utc=datetime.now(UTC).isoformat(),
                  passed=all(c['passed'] for c in checks), manifest_sha256=digest(manifest_path),
                  policy_sha256=digest(policy_path), checks=checks, metrics=metrics,
                  estimated_daily_seconds=daily_seconds,
                  ensemble_report_sha256=digest(ensemble_report),
                  evidence={stage:json.loads((Path(path)/'run.json').read_text()) for stage,path in [('offline',offline),('replay',replay),('shadow',shadow)]})
    # Infinity is useful internally for fail-closed checks, but not valid report JSON.
    def clean(value):
        if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, (float,np.floating)) and not np.isfinite(value): return None
        return value
    output=staging_path(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    return report
