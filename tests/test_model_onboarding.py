from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

from rt_forecast_dashboard.model_onboarding.contracts import frames, records, digest
from rt_forecast_dashboard.model_onboarding.process import ModelProcess
from rt_forecast_dashboard.model_onboarding.runner import run
from rt_forecast_dashboard.model_onboarding.gates import evaluate
from rt_forecast_dashboard.model_onboarding.registry import check_registry
from rt_forecast_dashboard.model_onboarding.datasets import combine_runs

ROOT = Path(__file__).resolve().parents[1]


def window(day='2024-01-01', hours=30):
    from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp
    origin = brussels_cutoff_timestamp(pd.Timestamp(day).date())
    context = pd.DataFrame({'timestamp':pd.date_range(end=origin-pd.Timedelta(hours=1),periods=hours,freq='h'), 'actual_mw':100.})
    future = pd.DataFrame({'timestamp':pd.date_range(origin,periods=30,freq='h'), 'actual_mw':1e12})
    return dict(zone='BE',target='load',run_date=day,origin_utc=origin.isoformat(),
                inputs_available_at_utc=origin.isoformat(),covariates=[],
                context=records(context),future=records(future),actuals=[101.]*30)


def request():
    w=window()
    return dict(context=w['context'],future=w['future'],covariates=[],target='load',context_hours=30,horizon=30)


def worker(behavior=None, **kwargs):
    return ModelProcess(sys.executable,'tests.fixtures.onboarding_adapter:create_adapter',{'behavior':behavior},
                        startup_timeout=10,timeout=.2,**kwargs)


def test_contract_strips_future_actuals_and_checks_grid():
    w=window()
    past,future=frames(w['context'],w['future'],[],context_hours=30)
    assert 'actual_mw' not in future
    assert past.actual_mw.eq(100).all()
    w['context'][-1]['timestamp']=w['future'][0]['timestamp']
    with pytest.raises(ValueError): frames(w['context'],w['future'],[],context_hours=30)


def test_worker_cannot_mutate_inputs_or_read_upstream_secrets(monkeypatch):
    monkeypatch.setenv('ENTSOE_API_KEY','test-secret-not-for-candidate')
    w=worker()
    r=request()
    try:
        first,cold=w.predict(r)
        second,warm=w.predict(r)
        assert cold and not warm
        assert first==second==[100.]*30
        assert r['context'][0]['actual_mw']==100
        assert 'mutated' not in r['future'][0]
    finally: w.close()


@pytest.mark.parametrize('behavior,match',[('network','Network disabled'),('bad_shape','finite'),('crash',None)])
def test_failed_worker_does_not_break_next_model(behavior,match):
    bad=worker(behavior)
    with pytest.raises(Exception,match=match): bad.predict(request())
    assert bad.process is None
    good=worker()
    try: assert good.predict(request())[0]==[100.]*30
    finally: good.close()


def test_timeout_kills_worker_and_memory_budget_is_enforced():
    w=worker('hang')
    try:
        w.predict(request())
        with pytest.raises(TimeoutError): w.predict(request())
        assert w.process is None
    finally: w.close()
    tiny=worker(memory_mb=1)
    with pytest.raises((MemoryError,RuntimeError)): tiny.predict(request())
    assert tiny.process is None


@pytest.fixture
def evidence(tmp_path):
    policy=yaml.safe_load((ROOT/'config/model_onboarding.yml').read_text())
    policy.update(required_system="any",context_hours=30,minimum_runs_per_year=2,minimum_months_per_year=1,minimum_runs_per_month=1,
                  minimum_warm_samples=1,minimum_shadow_days=2,maximum_warm_p95_seconds=2,
                  maximum_cold_seconds=10,maximum_daily_seconds=20)
    pp=tmp_path/'policy.yml';pp.write_text(yaml.safe_dump(policy))
    manifest=yaml.safe_load((ROOT/'examples/model_onboarding/manifest.yml').read_text())
    manifest.update(revision='a'*40,checkpoint='test/model',adapter_config={'model_id':'test/model','revision':'a'*40,'device':'cpu'},
                    adapter_factory='tests.fixtures.onboarding_adapter:create_adapter',adapter_source='tests/fixtures/onboarding_adapter.py')
    manifest['reference_models']=['reference_a','reference_b']
    for entry in manifest['licenses'].values(): entry.update(url='https://example.test/license',commercial_use='allowed',conditions_met=True)
    manifest['review']=dict(reviewer='Test fixture',evidence_url='https://example.test/review',training_data_overlap_assessment='Synthetic unit tests only')
    manifest['operations']={k:True for k in ('ensemble_check_passed','failure_isolation_check_passed','rollback_tested')}
    manifest['operations']['evidence_url']='https://example.test/test'
    import importlib.metadata
    from rt_forecast_dashboard.model_onboarding.environment import canonical
    active={}
    vcs={}
    for dist in importlib.metadata.distributions():
        if dist.metadata['Name']:
            name=canonical(dist.metadata['Name'])
            if name not in active:
                active[name]=dist.version
                direct=dist.read_text('direct_url.json')
                if direct:
                    revision=json.loads(direct).get('vcs_info',{}).get('commit_id')
                    if revision: vcs[name]=revision
    lock_path=tmp_path/'runtime.lock'
    lock_path.write_text(''.join((f'{name} @ git+https://example.test/repo@{vcs[name]}\n' if name in vcs else f'{name}=={version}\n') for name,version in sorted(active.items())))
    manifest['requirements_lock']=str(lock_path)
    mp=tmp_path/'manifest.yml';mp.write_text(yaml.safe_dump(manifest))
    baseline_paths={}
    for stage,days in [('offline',['2024-01-01','2024-01-02','2025-01-01','2025-01-02']),
                       ('replay',['2026-07-24','2026-07-25']),('shadow',['2026-10-19','2026-10-20'])]:
        windows=tmp_path/(stage+'.jsonl')
        windows.write_text(''.join(json.dumps(window(day))+'\n' for day in days))
        out=tmp_path/stage
        assert run(mp,windows,out,sys.executable,pp,stage)
        if stage=='shadow':
            # Synthetic evidence for unit testing, NOT a real seven-day shadow run.
            r=pd.read_csv(out/'runtime.csv')
            r['observed_at_utc']=(pd.to_datetime(r.origin_utc,utc=True)+pd.Timedelta(minutes=30)).astype(str)
            r.to_csv(out/'runtime.csv',index=False)
            meta=json.loads((out/'run.json').read_text());meta['runtime_sha256']=digest(out/'runtime.csv')
            (out/'run.json').write_text(json.dumps(meta))
        b=pd.read_csv(out/'forecasts.csv')
        refs=[]
        for name,value in [('reference_a',103.),('reference_b',104.)]:
            p=b.copy();p['model']=name;p['forecast_mw']=value;refs.append(p)
        bp=tmp_path/(stage+'-references.csv');pd.concat(refs).to_csv(bp,index=False)
        baseline_paths[stage]=bp
    from rt_forecast_dashboard.model_onboarding.ensemble import ensemble_check
    ensemble_check(mp,tmp_path/'offline',baseline_paths['offline'],pp,tmp_path/'ensemble.json')
    return tmp_path,mp,pp,manifest,baseline_paths


def admission(evidence):
    root,mp,pp,manifest,b=evidence
    return evaluate(mp,pp,offline=root/'offline',baselines=b['offline'],replay=root/'replay',
                    replay_baselines=b['replay'],shadow=root/'shadow',ensemble_report=root/'ensemble.json',output=root/'admission.json')


def test_gate_accepts_strong_matched_candidate_and_rejects_year_regression(evidence):
    root,mp,pp,manifest,b=evidence
    assert admission(evidence)['passed']
    path=root/'offline/forecasts.csv'
    f=pd.read_csv(path);f.loc[f.run_date.str.startswith('2025'),'forecast_mw']=200
    f.to_csv(path,index=False)
    meta=json.loads((root/'offline/run.json').read_text());meta['forecasts_sha256']=digest(path)
    (root/'offline/run.json').write_text(json.dumps(meta))
    report=admission(evidence)
    assert not report['passed']
    assert any('2025_nonregression' in c['name'] and not c['passed'] for c in report['checks'])


def test_gate_rejects_missing_mock_windows_and_unapproved_license(evidence):
    root,mp,pp,manifest,b=evidence
    f=pd.read_csv(root/'replay/forecasts.csv').iloc[:30]
    f.to_csv(root/'replay/forecasts.csv',index=False)
    report=admission(evidence)
    assert not report['passed']
    assert any('alignment' in c['name'] and not c['passed'] for c in report['checks'])
    manifest['licenses']['weights']['commercial_use']='prohibited';mp.write_text(yaml.safe_dump(manifest))
    assert any(c['name']=='license_weights' and not c['passed'] for c in admission(evidence)['checks'])


def test_registry_blocks_unreviewed_new_model_and_stale_adapter(evidence):
    root,mp,pp,manifest,b=evidence
    base=root/'base.yml';base.write_text('models: {}\n')
    current=root/'registry.yml'
    spec=dict(enabled=True,required_for_completeness=False,adapter='isolated',adapter_factory=manifest['adapter_factory'],adapter_config=manifest['adapter_config'],
              python=sys.executable,timeout_seconds=2,startup_timeout_seconds=10,memory_mb=4096,threads=2,scope=manifest['scope'],targets=['load'],onboarding_manifest=str(mp),onboarding_report=str(root/'admission.json'))
    current.write_text(yaml.safe_dump({'models':{manifest['model_key']:spec}}))
    assert check_registry(current,base,pp)
    assert admission(evidence)['passed']
    assert not check_registry(current,base,pp)
    report=json.loads((root/'admission.json').read_text());report['evidence']['offline']['adapter_source_sha256']='changed'
    (root/'admission.json').write_text(json.dumps(report))
    assert check_registry(current,base,pp)


def test_combining_shadow_evidence_rejects_duplicate_windows(evidence):
    root,*_=evidence
    with pytest.raises(ValueError,match='Overlapping'):
        combine_runs([root/'shadow',root/'shadow'],root/'combined')


def test_publication_is_explicit_preserves_incumbents_and_rejects_offline(evidence):
    from rt_forecast_dashboard.model_onboarding.publish import publish
    from rt_forecast_dashboard.storage import ForecastStore
    root,mp,pp,manifest,b=evidence
    assert admission(evidence)['passed']
    config=dict(enabled=True,required_for_completeness=False,adapter='isolated',adapter_factory=manifest['adapter_factory'],adapter_config=manifest['adapter_config'],
                python=sys.executable,timeout_seconds=2,startup_timeout_seconds=10,memory_mb=4096,threads=2,scope=manifest['scope'],targets=['load'],onboarding_manifest=str(mp),onboarding_report=str(root/'admission.json'))
    registry=root/'registry.yml';registry.write_text(yaml.safe_dump({'models':{manifest['model_key']:config}}))
    with pytest.raises(ValueError,match='never offline'):
        publish(mp,root/'offline',registry,pp,root/'production')
    store=ForecastStore(data_dir=root/'production')
    f=pd.read_csv(root/'replay/forecasts.csv');f['run_at']='2026-07-24T18:30:00Z';f['model']='incumbent';f['forecast_mw']=999.
    for day,part in f.groupby('run_date'): store.append_forecasts(part,day)
    assert publish(mp,root/'replay',registry,pp,root/'production')==60
    saved=store.read_forecasts()
    assert saved[saved.model.eq('incumbent')].forecast_mw.eq(999).all()
    assert saved[saved.model.eq(manifest['model_key'])].source.eq('historical_mock_replay').all()


def test_future_model_scope_is_shared_by_inputs_completeness_ensemble_and_ui(monkeypatch):
    from rt_forecast_dashboard import covariates
    from rt_forecast_dashboard.pipeline import check_forecast_complete as complete
    from rt_forecast_dashboard.pipeline import build_selected_ensembles as ensemble
    from rt_forecast_dashboard.ui import analytics
    spec=dict(enabled=True,label='Future TSFM',targets=['load'],required_for_completeness=False,
              scope=[dict(zone='BE',target='load',covariates=[])])
    registry={'future':spec}
    for module in (covariates,complete,ensemble,analytics):
        monkeypatch.setattr(module,'model_registry',lambda:registry)
    assert covariates.covariates_for('load','BE','future')[1]==[]
    assert not complete.expected_groups()
    assert complete.expected_groups({'future'})=={('BE','load','future')}
    assert ensemble._component_columns('load',False,'BE')==['future']
    assert ensemble._component_columns('load',False,'FR')==[]
    assert 'Future TSFM' in analytics.online_enabled_model_families('load')


def test_isolated_production_adapter_preserves_frames_and_reuses_worker():
    from rt_forecast_dashboard.models.isolated import IsolatedAdapter
    w=window(hours=2208)
    past,future=pd.DataFrame(w['context']),pd.DataFrame(w['future'])
    past_before,future_before=past.copy(deep=True),future.copy(deep=True)
    config=dict(python=sys.executable,adapter_factory='tests.fixtures.onboarding_adapter:create_adapter',
                adapter_config={},startup_timeout_seconds=10)
    a,b=IsolatedAdapter(config),IsolatedAdapter(config)
    try:
        assert len(a.predict(future,'load',past,[]))==30
        assert len(b.predict(future,'load',past,[]))==30
        assert a.worker is b.worker
        pd.testing.assert_frame_equal(past,past_before)
        pd.testing.assert_frame_equal(future,future_before)
    finally:
        if a.worker: a.worker.close()


def test_exported_snapshot_keeps_truth_outside_worker(tmp_path):
    from rt_forecast_dashboard.model_onboarding.datasets import export_windows
    w=window(hours=2208)
    pd.DataFrame(w['context']).to_parquet(tmp_path/'context.parquet')
    pd.DataFrame(w['future']).to_parquet(tmp_path/'future.parquet')
    pd.DataFrame([{**{k:w[k] for k in ('zone','target','run_date','origin_utc','inputs_available_at_utc')},
                   'context_path':'context.parquet','future_path':'future.parquet','covariates':'[]'}]).to_csv(tmp_path/'plan.csv',index=False)
    export_windows(tmp_path/'plan.csv',tmp_path/'windows.jsonl')
    row=json.loads((tmp_path/'windows.jsonl').read_text())
    assert row['actuals']==[1e12]*30
    assert 'actual_mw' not in row['future'][0]
    assert len(row['sources']['context_path'])==64


def test_review_can_be_finalized_without_replaying_identical_inference(evidence):
    root,mp,pp,manifest,b=evidence
    manifest['review']['reviewer']='Second operator'
    manifest['operations']['evidence_url']='https://example.test/final-review'
    mp.write_text(yaml.safe_dump(manifest))
    assert admission(evidence)['passed']


def test_gate_rejects_slow_models_even_when_predictions_are_good(evidence):
    root,mp,pp,manifest,b=evidence
    r=pd.read_csv(root/'offline/runtime.csv')
    r.loc[~r.cold_start.astype(str).str.lower().eq('true'),'elapsed_seconds']=100
    r.to_csv(root/'offline/runtime.csv',index=False)
    meta=json.loads((root/'offline/run.json').read_text());meta['runtime_sha256']=digest(root/'offline/runtime.csv')
    (root/'offline/run.json').write_text(json.dumps(meta))
    report=admission(evidence)
    assert not report['passed']
    assert any(c['name']=='warm_BE_load_p95' and not c['passed'] for c in report['checks'])


def test_invalid_reference_values_cannot_make_a_candidate_look_good(evidence):
    root,mp,pp,manifest,b=evidence
    frame=pd.read_csv(b['offline']);frame['forecast_mw']=np.inf
    frame.to_csv(b['offline'],index=False)
    with pytest.raises(ValueError,match='Invalid forecast'):
        admission(evidence)


def test_dependency_lock_must_match_installed_versions_and_cover_transitives():
    from rt_forecast_dashboard.model_onboarding.environment import locked_environment
    info=dict(active_versions={'numpy':'2.3.3','pandas':'2.3.3'},source_revisions={})
    assert not locked_environment(['numpy==2.3.3'],info)
    assert not locked_environment(['numpy==1.26.0','pandas==2.3.3'],info)
    assert locked_environment(['numpy==2.3.3','pandas==2.3.3'],info)
    info['source_revisions']['pandas']='a'*40
    assert not locked_environment(['numpy==2.3.3','pandas==2.3.3'],info)
    assert locked_environment(['numpy==2.3.3','pandas @ git+https://example.test/repo@'+'a'*40],info)


def test_staging_cannot_overwrite_configured_production_data(tmp_path,monkeypatch):
    from rt_forecast_dashboard.model_onboarding.contracts import staging_path
    monkeypatch.setenv('FORECAST_DATA_DIR',str(tmp_path/'live'))
    with pytest.raises(ValueError,match='cannot write production'):
        staging_path(tmp_path/'live/dashboard/forecasts_recent.parquet')
    assert staging_path(tmp_path/'staging')==tmp_path/'staging'
