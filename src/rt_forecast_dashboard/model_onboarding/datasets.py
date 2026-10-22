"""Portable frozen-window export and combination of separately recorded runs."""
import json
from pathlib import Path
import pandas as pd
from .contracts import digest, frames, records, staging_path


def export_windows(plan_path, output, context_hours=2208, horizon=None):
    plan_path = Path(plan_path)
    plan = pd.read_csv(plan_path, keep_default_na=False)
    if plan.duplicated(['zone','target','run_date']).any():
        raise ValueError('Duplicate plan entries')
    with staging_path(output).open('x') as handle:
        for row in plan.to_dict('records'):
            def read(name):
                path = plan_path.parent/row[name]
                return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
            context, future = read('context_path'), read('future_path')
            covs = json.loads(row['covariates'])
            past, next_ = frames(context,future,covs,context_hours=context_hours,horizon=horizon)
            actuals = future.actual_mw.where(future.actual_mw.notna(), None).tolist() if 'actual_mw' in future else [None]*len(next_)
            actuals = [None if pd.isna(x) else float(x) for x in actuals]
            result = {k:row[k] for k in ('zone','target','run_date','origin_utc','inputs_available_at_utc')}
            result.update(context=records(past), future=records(next_), covariates=covs, actuals=actuals,
                          sources={name:digest(plan_path.parent/row[name]) for name in ('context_path','future_path')})
            handle.write(json.dumps(result, allow_nan=False)+'\n')


def combine_runs(directories, output):
    out = staging_path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((Path(d)/'run.json').read_text()) for d in directories]
    for key in ('stage','model_key','revision','inference_sha256','policy_sha256','requirements_sha256','adapter_source_sha256','threads'):
        if len({m[key] for m in metas}) != 1:
            raise ValueError(f'Incompatible runs: {key}')
    if len({json.dumps(m.get('worker',{}).get('packages',[]),sort_keys=True) for m in metas})!=1:
        raise ValueError('Candidate environments differ across shards')
    for filename,keys in [('forecasts.csv',['zone','target','run_date','timestamp','model']),('runtime.csv',['zone','target','run_date'])]:
        for d,m in zip(directories,metas):
            if digest(Path(d)/filename)!=m[filename.split('.')[0]+'_sha256']:
                raise ValueError('Run artifact fingerprint changed')
        frame = pd.concat([pd.read_csv(Path(d)/filename) for d in directories], ignore_index=True)
        if frame.duplicated(keys).any():
            raise ValueError('Overlapping runs must not be counted twice')
        frame.to_csv(out/filename,index=False)
    meta = dict(metas[0], completed=all(m.get('completed') for m in metas), combined_runs=metas, forecasts_sha256=digest(out/'forecasts.csv'),runtime_sha256=digest(out/'runtime.csv'))
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
