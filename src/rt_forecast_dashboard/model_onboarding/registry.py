"""PR guard for new/enabled/revised production model registrations."""
import json
from pathlib import Path
import yaml
from .contracts import digest


CRITICAL = ('adapter','checkpoint_path','model_id','revision','adapter_factory','adapter_config','python','targets','scope','timeout_seconds','startup_timeout_seconds','memory_mb','threads')


def check_registry(current_path, base_path, policy_path):
    current = yaml.safe_load(Path(current_path).read_text())['models']
    base = yaml.safe_load(Path(base_path).read_text())['models']
    policy=yaml.safe_load(Path(policy_path).read_text())
    errors = []
    for key, config in current.items():
        previous = base.get(key, {})
        changed = not previous.get('enabled', False) or any(config.get(k)!=previous.get(k) for k in CRITICAL)
        if not config.get('enabled', False) or config.get('pipeline_generated') or (not changed and not config.get('onboarding_manifest')):
            continue
        try:
            manifest_path = Path(config['onboarding_manifest'])
            report_path = Path(config['onboarding_report'])
            manifest = yaml.safe_load(manifest_path.read_text())
            report = json.loads(report_path.read_text())
            if manifest['model_key'] != key or report['model_key'] != key:
                raise ValueError('Model identity mismatch')
            if config.get('required_for_completeness') is not False:
                raise ValueError('New models must remain optional for core-job completeness; their own job validates them explicitly')
            if config.get('adapter') != 'isolated':
                raise ValueError('New models must use the bounded isolated adapter; exceptions need a policy/code review')
            if config.get('adapter_factory')!=manifest.get('adapter_factory') or config.get('adapter_config')!=manifest.get('adapter_config'):
                raise ValueError('Registered adapter does not match evaluated manifest')
            for field,limit in [('timeout_seconds','maximum_warm_p95_seconds'),('startup_timeout_seconds','maximum_cold_seconds'),('memory_mb','maximum_peak_rss_mb')]:
                if not isinstance(config.get(field),(int,float)) or not 0<config[field]<=policy[limit]:
                    raise ValueError(f'Registered {field} exceeds or omits evaluated policy budget')
            if config.get('threads')!=policy['threads']:
                raise ValueError('Registered threads differ from evaluated profile')
            if not config.get('python'):
                raise ValueError('Candidate interpreter missing')
            if config.get('scope') != manifest.get('scope'):
                raise ValueError('Registered scope/covariates differ from evaluated manifest')
            scope_targets = {s['target'] for s in manifest['scope']}
            if set(config.get('targets', [])) != scope_targets:
                raise ValueError('Registered targets exceed/differ from evaluated scope')
            if (report.get('passed') is not True or not report.get('checks')
                    or not all(c.get('passed') is True for c in report['checks'])
                    or report.get('manifest_sha256') != digest(manifest_path)
                    or report.get('policy_sha256') != digest(policy_path)):
                raise ValueError('Missing, failed or stale admission report')
            if any(e.get('adapter_source_sha256') != digest(manifest['adapter_source'])
                   or e.get('requirements_sha256') != digest(manifest['requirements_lock'])
                   for e in report.get('evidence', {}).values()):
                raise ValueError('Adapter source or dependency lock changed since evaluation')
            required_checks = {'valid_scope','cpu_profile','license_code','license_weights','license_input_data',
                               'immutable_revision','pinned_adapter_config','pinned_dependencies','human_review',
                               'ensemble_check_passed','failure_isolation_check_passed','rollback_tested',
                               'ensemble_performance','shadow_live_timing','cold_start_budget','memory_budget','daily_budget'}
            required_checks |= {stage+'_'+suffix for stage in ('offline','replay','shadow')
                                for suffix in ('completed','fingerprints','environment_lock','hardware','reliability','model','finite_predictions','complete_horizons')}
            if not required_checks.issubset({c['name'] for c in report['checks']}):
                raise ValueError('Admission report omits required checks')
            if set(report.get('evidence', {})) != {'offline','replay','shadow'}:
                raise ValueError('All three evaluated stages are required')
        except (KeyError, ValueError, OSError, TypeError) as exc:
            errors.append(f'{key}: {exc}')
    return errors
