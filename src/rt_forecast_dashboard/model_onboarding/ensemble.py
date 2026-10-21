"""Offline leave-one-model-out ensemble comparison; no production writes."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import yaml
from .contracts import digest, inference_digest, KEYS, staging_path
from .gates import _read_forecasts


def ensemble_check(manifest_path, candidate_dir, baselines_path, policy_path, output):
    from rt_forecast_dashboard.config import model_registry
    from rt_forecast_dashboard.pipeline.build_selected_ensembles import build_selected_ensembles
    from rt_forecast_dashboard.storage import ForecastStore
    spec=yaml.safe_load(Path(manifest_path).read_text())
    policy=yaml.safe_load(Path(policy_path).read_text())
    candidate_path=Path(candidate_dir)/'forecasts.csv'
    meta=json.loads((Path(candidate_dir)/'run.json').read_text())
    if (meta.get('inference_sha256')!=inference_digest(manifest_path)
            or meta.get('forecasts_sha256')!=digest(candidate_path) or meta.get('completed') is not True):
        raise ValueError('Stale/incomplete candidate evidence')
    candidate=_read_forecasts(candidate_path)
    baseline=_read_forecasts(baselines_path)
    if not candidate.model.eq(spec['model_key']).all() or baseline.model.eq(spec['model_key']).any():
        raise ValueError('Candidate and reference identities must be separate')
    scope={(s['zone'],s['target']) for s in spec['scope']}
    baseline=baseline[[pair in scope for pair in zip(baseline.zone,baseline.target)]].copy()
    registry=model_registry()
    for key,part in baseline.groupby('model'):
        if key not in registry:
            registry[key]=dict(enabled=True,label=key,targets=part.target.unique().tolist())
    registry[spec['model_key']]=dict(enabled=True,label=spec['model_key'],targets=sorted({t for _,t in scope}),scope=spec['scope'])
    def storage_fields(frame):
        frame=frame.copy()
        origin=frame.groupby(['zone','target','run_date']).timestamp.transform('min')
        frame['run_at']=origin
        frame['context_hours']=policy['context_hours']
        frame['context_start']=origin-pd.Timedelta(hours=policy['context_hours'])
        frame['context_end']=origin-pd.Timedelta(hours=1)
        return frame
    all_forecasts=pd.concat([baseline,candidate],ignore_index=True)
    actuals=all_forecasts.groupby(KEYS).actual_mw.agg(['min','max'])
    if not np.allclose(actuals['min'],actuals['max'],equal_nan=True):
        raise ValueError('Inconsistent actuals in ensemble comparison')
    actuals=actuals.reset_index()[['zone','target','timestamp','min']].drop_duplicates(['zone','target','timestamp']).rename(columns={'min':'actual_mw'})
    results={}
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'actuals.parquet';actuals.to_parquet(path,index=False)
        for label,frame in [('without',baseline),('with',all_forecasts)]:
            frame=storage_fields(frame)
            class PreviewStore(ForecastStore):
                def read_forecasts(self): return frame.copy()
                def append_forecasts(self,*args,**kwargs): pass
            selected=dict(registry)
            if label=='without': selected.pop(spec['model_key'])
            with redirect_stdout(io.StringIO()):
                results[label]=build_selected_ensembles(PreviewStore(data_dir=Path(temp)/label),actuals_path=path,
                    run_dates=sorted(candidate.run_date.unique()),model_registry_override=selected,
                    offline_weights_path=Path(temp)/'no-old-weights.csv')
    if any(f.empty for f in results.values()):
        raise ValueError('Not enough reference components to evaluate the ensemble')
    with_=results['with'];without=results['without']
    common=with_.merge(without,on=KEYS,suffixes=('_with','_without'),validate='one_to_one')
    common=common.dropna(subset=['actual_mw_with','actual_mw_without'])
    rows=[]
    for (zone,target),part in common.groupby(['zone','target']):
        if (zone,target) not in scope: continue
        for year,year_part in part.groupby(pd.to_datetime(part.run_date).dt.year):
            before=float((year_part.forecast_mw_without-year_part.actual_mw_without).abs().mean())
            after=float((year_part.forecast_mw_with-year_part.actual_mw_with).abs().mean())
            ratio=after/before if before>0 else (1. if after==0 else float('inf'))
            rows.append(dict(zone=zone,target=target,year=int(year),hours=len(year_part),
                             without_mae_mw=before,with_mae_mw=after,mae_ratio=ratio))
    included=with_.ensemble_components.str.split(',').map(lambda keys:spec['model_key'] in keys).all()
    # All declared scope/year combinations need observed comparisons.
    observed={(r['zone'],r['target'],r['year']) for r in rows}
    expected={(z,t,y) for z,t in scope for y in policy['offline_years']}
    report=dict(passed=bool(included and expected.issubset(observed) and rows
                            and all(r['mae_ratio']<=policy['maximum_year_mae_ratio'] for r in rows)),
                inference_sha256=inference_digest(manifest_path),candidate_forecasts_sha256=digest(candidate_path),
                baselines_sha256=digest(baselines_path),policy_sha256=digest(policy_path),metrics=rows)
    for row in rows:
        if not np.isfinite(row['mae_ratio']): row['mae_ratio']=None
    output=staging_path(output)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
