from datetime import date
from pathlib import Path
import io,zipfile
import numpy as np
import pandas as pd
import pytest
from scripts.select_de_solar_weather_points import entries,unit_record,select_points
from scripts.report_de_solar_weather_study import shared_windows,bootstrap_delta,LABELS
from scripts.de_solar_weather_study import expected_grid


def test_truncated_zip_directory_keeps_original_offsets():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('other.xml','abc'*1000);z.writestr('EinheitenSolar_1.xml','test')
    data=stream.getvalue();tail=data[-300:]
    result=entries(tail,len(data));assert len(result)==1
    assert data[result[0]['offset']:result[0]['offset']+4]==b'PK\x03\x04'


def test_private_rooftop_coordinates_use_coarse_postcode_not_drop_unit():
    values=dict(Land='84',EinheitSystemstatus='472',Nettonennleistung='10',Bruttoleistung='12',Postleitzahl='01067',
                EinheitBetriebsstatus='35',Registrierungsdatum='2020-01-01',Inbetriebnahmedatum='2020-01-01',AnlagenbetreiberMastrNummer='private')
    row=unit_record(values,{'01067':(51.05,13.74)})
    assert row['weight']==.01 and row['coordinate_source']=='postcode' and row['lat']==51.05
    assert 'AnlagenbetreiberMastrNummer' not in row
    assert unit_record({**values,'Land':'85'},{'01067':(51.05,13.74)}) is None


def test_capacity_points_are_deterministic_four_grid_locations():
    d=pd.DataFrame(dict(latitude=[48.,48.1,50.,50.1,52.,52.1,54.,54.1],longitude=[10.,10.1,7.,7.1,13.,13.1,9.,9.1],net_mw=[20.,30.]*4,units=[10]*8))
    a=select_points(d);b=select_points(d)
    pd.testing.assert_frame_equal(a,b);assert len(a)==4 and a.cluster_share.sum()==pytest.approx(1)
    assert a[['gfs_lat','gfs_lon']].drop_duplicates().shape[0]==4
    assert np.allclose(a.gfs_lat/.1171875,np.round(a.gfs_lat/.1171875))


def test_shared_scores_reject_partial_dst_window_and_missing_truth():
    day='2026-10-24';grid=expected_grid(day);assert len(grid)==31
    rows=[dict(run_date=day,timestamp=t,model=model,forecast_mw=1.,actual_mw=2.,tso_forecast_mw=3.) for model in LABELS for t in grid]
    a=pd.DataFrame(rows);b=a.copy();assert shared_windows({'baseline':a,'candidate':b})==[day]
    b=b.iloc[:-1];assert shared_windows({'baseline':a,'candidate':b})==[]
    b=a.copy();b.loc[0,'actual_mw']=np.nan;assert shared_windows({'baseline':a,'candidate':b})==[]


def test_paired_block_bootstrap_preserves_identical_control():
    d=pd.DataFrame(dict(run_date=pd.date_range('2026-07-24',periods=30).astype(str),baseline_abs_sum=np.arange(30.)+100,
                        candidate_abs_sum=np.arange(30.)+100,hours=[30]*30))
    assert np.array_equal(bootstrap_delta(d),[0.,0.])
    d['candidate_abs_sum']-=30
    assert np.allclose(bootstrap_delta(d),[-1.,-1.])


def test_replay_preserves_saved_tso_and_rejects_future_truth(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import de_solar_weather_study as study
    day='2026-07-24';grid=expected_grid(day)
    pd.DataFrame(dict(run_date=day,timestamp=grid,tso_forecast_mw=np.arange(len(grid))+300.)).to_parquet(tmp_path/'online_tso_by_origin.parquet')
    context=pd.DataFrame(dict(timestamp=pd.date_range(grid[0]-pd.Timedelta(hours=2),periods=2,freq='h'),actual_mw=[1.,2.],tso_forecast_mw=[2.,3.]))
    future=pd.DataFrame(dict(timestamp=grid,tso_forecast_mw=[999.]*len(grid)))
    quality=SimpleNamespace(missing_hours=0,imputed_hours=0,methods='none')
    monkeypatch.setattr(study,'prepare_scope',lambda *a:({day:(context.copy(),future.copy(),quality,np.ones(len(grid)))},[]))
    prepared,days=study.prepare_arms(tmp_path,day,day)
    assert days==[day]
    for arm in prepared:
        c,f,q,t=prepared[arm][day]
        assert np.array_equal(f.tso_forecast_mw,np.arange(len(grid))+300.)
        assert 'actual_mw' not in f and c.timestamp.max()<f.timestamp.min()
    future['actual_mw']=1.
    with pytest.raises(AssertionError):study.prepare_arms(tmp_path,day,day)


def test_changed_online_configuration_is_rejected_before_replay(tmp_path):
    import json
    from scripts.de_solar_weather_study import sha,validate_configuration
    config=tmp_path/'frozen.yml';config.write_text('horizon: 30\n')
    (tmp_path/'configuration_hashes.json').write_text(json.dumps({str(config):sha(config)}))
    validate_configuration(tmp_path)
    config.write_text('horizon: 24\n')
    with pytest.raises(RuntimeError,match='differs from frozen'):validate_configuration(tmp_path)
