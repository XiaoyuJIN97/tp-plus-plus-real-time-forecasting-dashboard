from pathlib import Path
import pandas as pd
import pytest
from scripts.stage_dk_solar_history_repair import stage
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import forecast_delivery_grid


def rows(model,hours=30,zone='DK1',prediction=90.):
    grid=forecast_delivery_grid('2026-08-05')[:hours]
    return pd.DataFrame(dict(run_date='2026-08-05',zone=zone,target='solar',model=model,
        model_label=model,run_at='2026-11-03T00:00:00Z',source='historical_mock_replay',
        context_hours=2208,context_end=grid[0]-pd.Timedelta(hours=1),timestamp=grid,
        actual_mw=100.,forecast_mw=prediction))


def setup(tmp_path,incoming_hours=30,existing_hours=24):
    source=ForecastStore(tmp_path/'source');replay=ForecastStore(tmp_path/'replay')
    source.append_forecasts(pd.concat([rows('t0_gamma_online',existing_hours,prediction=80.),
        rows('timesfm3_online',prediction=70.),rows('t0_gamma_online',zone='BE',prediction=60.)]),'2026-08-05')
    replay.append_forecasts(rows('t0_gamma_online',incoming_hours),'2026-08-05')
    plan=tmp_path/'plan.csv';rows('t0_gamma_online')[['run_date','zone','target','model']].drop_duplicates().to_csv(plan,index=False)
    for z in ['DK1','DK2']:
        p=tmp_path/'inputs/entsoe/data/hourly'/z/'actual_solar_generation/2026.parquet';p.parent.mkdir(parents=True)
        pd.DataFrame(dict(timestamp_utc=forecast_delivery_grid('2026-08-05'),value=110.)).to_parquet(p)
    pd.DataFrame(columns=['zone','timestamp_utc']).to_parquet(tmp_path/'recovered_published_zero_actuals.parquet')
    return source,replay,plan


def test_staging_repairs_only_incomplete_models_and_normalizes_scoped_truth(tmp_path):
    source,replay,plan=setup(tmp_path)
    report=stage(source.data_dir,replay.data_dir,plan,tmp_path/'inputs',tmp_path/'staged')
    assert report['imported_groups']==1
    result=ForecastStore(tmp_path/'staged').read_forecasts()
    assert len(result[result.zone.eq('DK1')&result.model.eq('t0_gamma_online')])==30
    assert result[result.zone.eq('DK1')&result.model.eq('timesfm3_online')].forecast_mw.eq(70).all()
    assert result[result.zone.eq('BE')].forecast_mw.eq(60).all()
    assert result[result.zone.eq('BE')].actual_mw.eq(100).all()
    assert result[result.zone.eq('DK1')].actual_mw.eq(110).all()
    assert len(source.read_forecasts())==84


def test_incomplete_replays_fail_before_any_staged_import(tmp_path):
    source,replay,plan=setup(tmp_path,incoming_hours=24)
    with pytest.raises(ValueError,match='invalid forecast groups'):
        stage(source.data_dir,replay.data_dir,plan,tmp_path/'inputs',tmp_path/'staged')
    assert not (tmp_path/'staged').exists()


def test_preexisting_complete_forecast_is_never_replaced(tmp_path):
    source,replay,plan=setup(tmp_path,existing_hours=30)
    report=stage(source.data_dir,replay.data_dir,plan,tmp_path/'inputs',tmp_path/'staged')
    assert report['imported_groups']==0
    result=ForecastStore(tmp_path/'staged').read_forecasts()
    assert result[result.zone.eq('DK1')&result.model.eq('t0_gamma_online')].forecast_mw.eq(80).all()


def test_published_dk_solar_history_keeps_full_windows_before_september():
    from rt_forecast_dashboard.time_utils import is_complete_forecast_window
    root=Path(__file__).resolve().parents[1]
    models={'tso_reference','persistence','chronos2_online','timesfm3_online',
            't0_gamma_online','timestral_online','xgboost_online','ensemble_selected'}
    for day in ['2026-07-24','2026-08-05','2026-09-01','2026-09-20','2026-10-24']:
        frame=pd.read_csv(root/'data/forecasts'/f'forecasts_{day}.csv')
        frame=frame[frame.zone.isin(['DK1','DK2'])&frame.target.eq('solar')&frame.model.isin(models)]
        assert set(zip(frame.zone,frame.model))=={(z,m) for z in ['DK1','DK2'] for m in models}
        for _,part in frame.groupby(['zone','model']):
            assert is_complete_forecast_window(part.timestamp,day)
            assert part.actual_mw.notna().all()
