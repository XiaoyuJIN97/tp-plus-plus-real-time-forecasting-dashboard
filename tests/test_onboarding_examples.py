"""Worked examples use the real window contract without downloading weights."""
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.onboarding_foundation_example import initialize, freeze
from scripts.onboarding_foundation_adapter import Adapter, CHECKPOINTS
from rt_forecast_dashboard.models.foundation_runtime import REVISIONS


def inputs(covs):
    past = pd.DataFrame({'timestamp':pd.date_range('2023-10-02',periods=2208,freq='h',tz='UTC'),
                         'actual_mw':100.})
    future = pd.DataFrame({'timestamp':pd.date_range(past.timestamp.iloc[-1]+pd.Timedelta(hours=1),periods=24,freq='h'),
                           'actual_mw':120.})
    for name in covs:
        past[name]=1.
        future[name]=2.
    return past,future


@pytest.mark.parametrize('family',list(CHECKPOINTS))
def test_native_tensor_contract(family,monkeypatch):
    import contextlib
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    adapter = Adapter.__new__(Adapter)
    adapter.family = family
    class Native:
        def predict(self,x,**kwargs):
            assert x.shape==(1,2208)
            assert kwargs['future_covariates'].shape==(1,1,2232)
            return types.SimpleNamespace(median=np.ones((1,24)))
        def forecast(self,x,**kwargs):
            assert x.shape==(1,1,2208)
            assert kwargs['known_covariates'].shape==(1,1,2232)
            return types.SimpleNamespace(median=np.ones((1,1,24)))
    adapter.model=Native()
    past,future=inputs(['weather'])
    assert adapter.predict(future,'load',past,['weather']).shape==(24,)
    assert CHECKPOINTS[family][1]==REVISIONS[family]


def test_generated_manifest_and_frozen_data(tmp_path):
    spec=initialize('t0_gamma',tmp_path/'model')
    assert all(x['commercial_use']=='unknown' for x in spec['licenses'].values())
    covs=spec['scope'][0]['covariates']
    past,future=inputs(covs)
    past.to_csv(tmp_path/'context.csv',index=False)
    future.to_csv(tmp_path/'future.csv',index=False)
    origin=future.timestamp.iloc[0]
    plan=pd.DataFrame([dict(zone='BE',target='load',run_date=str(origin.date()),origin_utc=origin.isoformat(),
        inputs_available_at_utc=origin.isoformat(),context_path='context.csv',future_path='future.csv')])
    plan.to_csv(tmp_path/'inventory.csv',index=False)
    refs=pd.concat([future.assign(zone='BE',target='load',run_date=str(origin.date()),model=m,forecast_mw=110.)
                    for m in spec['reference_models']],ignore_index=True)
    refs.to_csv(tmp_path/'refs.csv',index=False)
    freeze(tmp_path/'model/manifest.yml',tmp_path/'inventory.csv',tmp_path/'refs.csv',tmp_path/'frozen')
    import json
    window=json.loads((tmp_path/'frozen/windows.jsonl').read_text())
    assert len(window['context'])==2208 and len(window['future'])==24
    assert 'actual_mw' not in window['future'][0]
    assert window['actuals']==[120.]*24
    refs.iloc[:-1].to_csv(tmp_path/'missing.csv',index=False)
    with pytest.raises(ValueError,match='does not cover'):
        freeze(tmp_path/'model/manifest.yml',tmp_path/'inventory.csv',tmp_path/'missing.csv',tmp_path/'bad')
