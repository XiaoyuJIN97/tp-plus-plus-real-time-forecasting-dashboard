"""Generate manifests or freeze already prepared, vintage-aware local data.

PYTHONPATH=src:. python scripts/onboarding_foundation_example.py --help
No downloads, production writes, synthetic scores or automatic approval.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from scripts.onboarding_foundation_adapter import CHECKPOINTS
from rt_forecast_dashboard.covariates import covariates_for
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp
from rt_forecast_dashboard.model_onboarding.contracts import frames, staging_path
from rt_forecast_dashboard.model_onboarding.datasets import export_windows


def initialize(family, output, zone='BE', target='load'):
    out = staging_path(output)
    out.mkdir(parents=True, exist_ok=False)
    key = family+'_onboarding_example'
    model_id, revision = CHECKPOINTS[family]
    covs = covariates_for(target, zone, family+'_online')[1]
    scope = [dict(zone=zone, target=target, covariates=covs)]
    config = dict(family=family, model_id=model_id, revision=revision, device='cpu',
                  snapshot_dir=str((out/'checkpoint').resolve()), local_files_only=True)
    spec = dict(model_key=key, checkpoint=model_id, revision=revision,
        requirements_lock=str(out/'requirements.lock'),
        adapter_source='scripts/onboarding_foundation_adapter.py',
        prepare_factory='scripts.onboarding_foundation_adapter:prepare_model',
        adapter_factory='scripts.onboarding_foundation_adapter:create_adapter',
        adapter_config=config, scope=scope,
        reference_models=['chronos2_online', 'ridge_3mo_context' if target=='load' else 'xgboost_online',
                          'timesfm3_online', 't0_gamma_online', 'timestral_online', 'tso_reference'],
        licenses={name:dict(url='', commercial_use='unknown', conditions_met=False)
                  for name in ('code','weights','input_data')},
        review=dict(reviewer='', evidence_url='', training_data_overlap_assessment=''),
        operations=dict(ensemble_check_passed=False, failure_isolation_check_passed=False,
                        rollback_tested=False, evidence_url=''))
    (out/'manifest.yml').write_text(yaml.safe_dump(spec, sort_keys=False))
    registration = dict(label=family+' onboarding example', enabled=False, adapter='isolated',
        adapter_factory=spec['adapter_factory'], adapter_config=config,
        python=f'.model-envs/{family}/bin/python', targets=[target], scope=scope,
        timeout_seconds=30, startup_timeout_seconds=180, memory_mb=4096, threads=2,
        required_for_completeness=False, onboarding_manifest=str(out/'manifest.yml'),
        onboarding_report=str(out/'admission.json'))
    (out/'registry-entry.yml').write_text(yaml.safe_dump({key:registration}, sort_keys=False))
    return spec


def freeze(manifest, inventory, references, output):
    """Inventory is metadata plus one context/future file pair per origin.

    No resampling/interpolation: use the upstream context-quality preparation
    first and retain that evidence. The inventory's availability is attested by
    the operator; this script cannot infer historical publication times.
    """
    spec = yaml.safe_load(Path(manifest).read_text())
    scopes = {(s['zone'],s['target']):s['covariates'] for s in spec['scope']}
    source = Path(inventory)
    plan = pd.read_csv(source, keep_default_na=False)
    required = {'zone','target','run_date','origin_utc','inputs_available_at_utc','context_path','future_path'}
    if required-set(plan):
        raise ValueError(f'Missing inventory fields: {required-set(plan)}')
    if plan.empty or plan.duplicated(['zone','target','run_date']).any():
        raise ValueError('Inventory must contain unique, nonempty daily windows')
    out = staging_path(output)
    if out.exists():
        raise ValueError('Choose a new staging output directory')
    # Validate everything before creating outputs.
    expected = []
    for row in plan.to_dict('records'):
        covs = scopes[(row['zone'],row['target'])]
        loaded = []
        for field in ('context_path','future_path'):
            path = (source.parent/row[field]).resolve()
            loaded.append(pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path))
        context, future = frames(*loaded,covs)
        origin = pd.Timestamp(row['origin_utc'])
        available = pd.Timestamp(row['inputs_available_at_utc'])
        if origin.tzinfo is None or available.tzinfo is None or available>origin:
            raise ValueError('Explicit timezone and input availability <= origin are required')
        if origin != future.timestamp.iloc[0] or origin != brussels_cutoff_timestamp(pd.Timestamp(row['run_date']).date()):
            raise ValueError('Origin/delivery/run_date mismatch')
        expected.extend((row['zone'],row['target'],str(row['run_date']),t) for t in future.timestamp)
    refs = pd.read_csv(references, dtype={'run_date':str})
    keys = ['zone','target','run_date','timestamp']
    required_refs = set(keys+['model','forecast_mw','actual_mw'])
    if required_refs-set(refs):
        raise ValueError(f'Missing reference fields: {required_refs-set(refs)}')
    refs.timestamp = pd.to_datetime(refs.timestamp,utc=True)
    window_index = pd.MultiIndex.from_tuples(expected,names=keys)
    refs = refs[refs.model.isin(spec['reference_models']) & pd.MultiIndex.from_frame(refs[keys]).isin(window_index)].copy()
    if refs.duplicated(keys+['model']).any():
        raise ValueError('Duplicate reference forecasts')
    for model in spec['reference_models']:
        if set(map(tuple,refs.loc[refs.model.eq(model),keys].to_numpy())) != set(expected):
            raise ValueError(f'Reference {model} does not cover every requested hour')
    out.mkdir(parents=True)
    plan['covariates'] = [json.dumps(scopes[(r.zone,r.target)]) for r in plan.itertuples()]
    for field in ('context_path','future_path'):
        plan[field] = [(source.parent/p).resolve().as_posix() for p in plan[field]]
    plan.to_csv(out/'plan.csv', index=False)
    export_windows(out/'plan.csv',out/'windows.jsonl')
    refs[list(keys)+['model','forecast_mw','actual_mw']].to_csv(out/'references.csv',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--family',choices=list(CHECKPOINTS),required=True)
    init.add_argument('--output',required=True)
    init.add_argument('--zone',default='BE')
    init.add_argument('--target',default='load',choices=['load','solar','wind_onshore','wind_offshore'])
    data = sub.add_parser('freeze')
    for name in ('manifest','inventory','references','output'):
        data.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.command=='init':
        initialize(args.family,args.output,args.zone,args.target)
    else:
        freeze(args.manifest,args.inventory,args.references,args.output)


if __name__=='__main__':
    main()
