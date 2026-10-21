"""Explicit, admitted-model-only import; never replaces incumbent model groups."""
from datetime import UTC, datetime
import json
from pathlib import Path
import pandas as pd
import yaml
from .contracts import digest, inference_digest
from .gates import _read_forecasts
from .registry import check_registry
from .process import ROOT


def publish(manifest_path, directory, registry_path, policy_path, data_dir):
    from rt_forecast_dashboard.storage import ForecastStore
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    registry = yaml.safe_load(Path(registry_path).read_text())['models']
    key = manifest['model_key']
    config = registry.get(key, {})
    if not config.get('enabled') or config.get('onboarding_manifest') != str(manifest_path):
        raise ValueError('Candidate must be admitted and enabled in registry before publication')
    # Revalidate evaluated configuration against an empty base; legacy entries are
    # filtered out so this command cannot accidentally clear another model.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        current = Path(temp)/'registry.yml'; base = Path(temp)/'base.yml'
        current.write_text(yaml.safe_dump({'models':{key:config}}));base.write_text('models: {}\n')
        errors = check_registry(current,base,policy_path)
        if errors: raise ValueError('; '.join(errors))
    directory = Path(directory)
    meta = json.loads((directory/'run.json').read_text())
    if meta.get('inference_sha256')!=inference_digest(manifest_path):
        raise ValueError('Candidate inference identity changed')
    policy = yaml.safe_load(Path(policy_path).read_text())
    for name,path in [('policy_sha256',policy_path),
                      ('requirements_sha256',ROOT/manifest['requirements_lock']),
                      ('adapter_source_sha256',ROOT/manifest['adapter_source']),
                      ('forecasts_sha256',directory/'forecasts.csv'),('runtime_sha256',directory/'runtime.csv')]:
        if meta.get(name)!=digest(path): raise ValueError(f'Stale run artifact: {name}')
    if meta.get('stage') not in {'replay','shadow'} or meta.get('model_key')!=key:
        raise ValueError('Only this model\'s replay/shadow results can be published; never offline tests')
    frame = _read_forecasts(directory/'forecasts.csv')
    scope = {(s['zone'],s['target']) for s in manifest['scope']}
    if set(zip(frame.zone,frame.target))-scope or not frame.model.eq(key).all():
        raise ValueError('Forecasts outside admitted model/scope')
    runtime = pd.read_csv(directory/'runtime.csv')
    if meta['stage']=='shadow':
        observed=pd.to_datetime(runtime.observed_at_utc,utc=True)
        origin=pd.to_datetime(runtime.origin_utc,utc=True)
        if not ((observed>=origin)&(observed<origin+pd.Timedelta(hours=12))).all():
            raise ValueError('Retrospective runs must not be published as online')
        live_expected={(z,t,str(day)) for z,t in scope for day in runtime.run_date.astype(str).unique()}
        if live_expected!=set(zip(runtime.zone,runtime.target,runtime.run_date.astype(str))):
            raise ValueError('Daily publication requires the full admitted scope')
    success = runtime.success.astype(str).str.lower().eq('true')
    expected = set(zip(runtime.zone,runtime.target,runtime.run_date.astype(str)))
    actual = set(zip(frame.zone,frame.target,frame.run_date))
    if not success.all() or actual!=expected or frame.empty:
        raise ValueError('Do not publish incomplete/failed candidate runs')
    for _,part in frame.groupby(['zone','target','run_date']):
        timestamps = part.timestamp.sort_values()
        if len(part)!=policy['horizon_hours'] or not timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError('Incomplete/non-hourly delivery window')
    from .contracts import prediction
    prediction(frame.forecast_mw.to_numpy(),len(frame))
    frame['run_at'] = datetime.now(UTC).isoformat()
    frame['source'] = 'historical_mock_replay' if meta['stage']=='replay' else 'online'
    frame['model_label'] = config.get('label',key)
    frame['context_hours'] = policy['context_hours']
    group = frame.groupby(['zone','target','run_date']).timestamp
    frame['context_end'] = group.transform('min')-pd.Timedelta(hours=1)
    frame['context_start'] = group.transform('min')-pd.Timedelta(hours=policy['context_hours'])
    frame['tso_forecast_mw'] = float('nan')
    frame['covariate_case'] = 'admitted_isolated_snapshot'
    store = ForecastStore(data_dir=Path(data_dir))
    for day,part in frame.groupby('run_date'):
        store.append_forecasts(part,str(day),replace_run=False)
    return len(frame)
