from __future__ import annotations
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import yaml

from .contracts import digest, frames, prediction, records, inference_digest, staging_path
from .process import ModelProcess, ROOT


def run(manifest_path, windows_path, output, python, policy_path, stage):
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    policy = yaml.safe_load(Path(policy_path).read_text())
    output = staging_path(output)
    if output.is_relative_to(ROOT/'data') or (output.exists() and any(output.iterdir())):
        raise ValueError('Use a new staging directory outside production data/')
    output.mkdir(parents=True, exist_ok=True)
    worker = ModelProcess(python, manifest['adapter_factory'], manifest.get('adapter_config', {}),
                          timeout=policy['maximum_warm_p95_seconds'], startup_timeout=policy['maximum_cold_seconds'],
                          memory_mb=policy['maximum_peak_rss_mb'], threads=policy['threads'], log_path=output/'worker.log')
    scope = {(s['zone'], s['target']):s['covariates'] for s in manifest['scope']}
    seen = set()
    runtime = []
    completed = False
    prediction_fields = ['zone','target','run_date','timestamp','model','forecast_mw','actual_mw']
    try:
        with Path(windows_path).open() as inputs, (output/'forecasts.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_fields)
            writer.writeheader()
            for line in inputs:
                row = json.loads(line)
                key = (row['zone'], row['target'], row['run_date'])
                if key in seen:
                    raise ValueError(f'Duplicate planned window: {key}')
                seen.add(key)
                start = time.monotonic()
                cold = worker.process is None
                entry = dict(zone=key[0], target=key[1], run_date=key[2], success=False, cold_start=cold,
                             elapsed_seconds=0., peak_rss_mb=0., error='',
                             origin_utc=row['origin_utc'], observed_at_utc=datetime.now(UTC).isoformat())
                try:
                    if key[:2] not in scope:
                        raise ValueError('Window outside manifest scope')
                    if row['covariates'] != scope[key[:2]]:
                        raise ValueError('Window covariates differ from evaluated manifest scope')
                    past, future = frames(row['context'], row['future'], row['covariates'],
                                          context_hours=policy['context_hours'], horizon=policy['horizon_hours'])
                    origin = pd.Timestamp(row['origin_utc'])
                    available = pd.Timestamp(row['inputs_available_at_utc'])
                    if origin.tzinfo is None or available.tzinfo is None or available > origin or origin != future.timestamp.min():
                        raise ValueError('Inputs must be documented as available by the forecast origin')
                    if str(origin.tz_convert('Europe/Brussels').date()) != key[2]:
                        raise ValueError('run_date must be the Brussels forecast-origin date')
                    truth = row.get('actuals', [None]*len(future))
                    if len(truth) != len(future):
                        raise ValueError('Actuals length does not match horizon')
                    values, actual_cold = worker.predict(dict(context=records(past), future=records(future),
                                                    covariates=row['covariates'], target=key[1],
                                                    context_hours=len(past), horizon=len(future)))
                    entry['cold_start'] = actual_cold
                    values = prediction(values, len(future))
                    for timestamp, value, actual in zip(future.timestamp, values, truth):
                        writer.writerow(dict(zone=key[0], target=key[1], run_date=key[2], timestamp=timestamp.isoformat(),
                                             model=manifest['model_key'], forecast_mw=value, actual_mw=actual))
                    handle.flush()
                    entry['success'] = True
                except Exception as exc:
                    entry['error'] = f'{type(exc).__name__}: {exc}'
                entry['elapsed_seconds'] = time.monotonic()-start
                entry['peak_rss_mb'] = worker.peak_rss_mb
                runtime.append(entry)
            completed = True
    finally:
        worker.close()
        pd.DataFrame(runtime, columns=['zone','target','run_date','success','cold_start','elapsed_seconds','peak_rss_mb','error','origin_utc','observed_at_utc']).to_csv(output/'runtime.csv', index=False)
        metadata = dict(completed=completed, stage=stage, model_key=manifest['model_key'], revision=manifest['revision'],
                        generated_at_utc=datetime.now(UTC).isoformat(), host_platform=platform.platform(),
                        threads=policy['threads'], worker=worker.info, manifest_sha256=digest(manifest_path), inference_sha256=inference_digest(manifest_path),
                        policy_sha256=digest(policy_path), windows_sha256=digest(windows_path),
                        requirements_sha256=digest(ROOT/manifest['requirements_lock']),
                        adapter_source_sha256=digest(ROOT/manifest['adapter_source']),
                        forecasts_sha256=digest(output/'forecasts.csv'), runtime_sha256=digest(output/'runtime.csv'))
        (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return all(row['success'] for row in runtime) and bool(runtime)
