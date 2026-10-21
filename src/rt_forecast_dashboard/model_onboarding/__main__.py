import argparse
import json
from .runner import run
from .gates import evaluate
from .registry import check_registry
from .datasets import export_windows, combine_runs
from .publish import publish
from .prepare import prepare
from .ensemble import ensemble_check


def main():
    parser = argparse.ArgumentParser(description='Stage and evaluate candidates; publication requires the explicit publish command')
    sub = parser.add_subparsers(dest='command', required=True)
    benchmark = sub.add_parser('run')
    for name in ('manifest','windows','output','python'):
        benchmark.add_argument('--'+name, required=True)
    benchmark.add_argument('--stage', required=True, choices=['offline','replay','shadow'])
    benchmark.add_argument('--policy', default='config/model_onboarding.yml')
    gate = sub.add_parser('gate')
    for name in ('manifest','offline','baselines','replay','replay-baselines','shadow','ensemble-report','output'):
        gate.add_argument('--'+name, required=True)
    gate.add_argument('--policy', default='config/model_onboarding.yml')
    registry = sub.add_parser('check-registry')
    registry.add_argument('--current', default='config/model_registry.yml')
    registry.add_argument('--base', required=True)
    registry.add_argument('--policy', default='config/model_onboarding.yml')
    export = sub.add_parser('export-windows')
    export.add_argument('--plan', required=True)
    export.add_argument('--output', required=True)
    combine = sub.add_parser('combine-runs')
    combine.add_argument('--directories', nargs='+', required=True)
    combine.add_argument('--output', required=True)
    importer = sub.add_parser('publish')
    importer.add_argument('--manifest', required=True)
    importer.add_argument('--run-directory', required=True)
    importer.add_argument('--registry', default='config/model_registry.yml')
    importer.add_argument('--policy', default='config/model_onboarding.yml')
    importer.add_argument('--data-dir', default='data')
    preparation = sub.add_parser('prepare')
    for name in ('manifest','python','output'):
        preparation.add_argument('--'+name,required=True)
    ensemble = sub.add_parser('ensemble-check')
    for name in ('manifest','candidate','baselines','output'):
        ensemble.add_argument('--'+name,required=True)
    ensemble.add_argument('--policy',default='config/model_onboarding.yml')
    args = parser.parse_args()
    if args.command=='run':
        passed = run(args.manifest,args.windows,args.output,args.python,args.policy,args.stage)
    elif args.command=='gate':
        report = evaluate(args.manifest,args.policy,offline=args.offline,baselines=args.baselines,
                          replay=args.replay,replay_baselines=args.replay_baselines,shadow=args.shadow,ensemble_report=args.ensemble_report,output=args.output)
        print(json.dumps({'passed':report['passed'],'failed_checks':[c for c in report['checks'] if not c['passed']]}, indent=2))
        passed = report['passed']
    elif args.command=='ensemble-check':
        passed=ensemble_check(args.manifest,args.candidate,args.baselines,args.policy,args.output)['passed']
    elif args.command=='prepare':
        prepare(args.manifest,args.python,args.output); passed=True
    elif args.command=='export-windows':
        export_windows(args.plan,args.output); passed=True
    elif args.command=='combine-runs':
        combine_runs(args.directories,args.output); passed=True
    elif args.command=='publish':
        print('Published rows:',publish(args.manifest,args.run_directory,args.registry,args.policy,args.data_dir)); passed=True
    else:
        errors = check_registry(args.current,args.base,args.policy)
        print('\n'.join(errors) if errors else 'No unapproved new production registrations')
        passed = not errors
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':
    main()
