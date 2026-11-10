import json
import pandas as pd
import pytest
from scripts.stage_weather_vintage_repair import stage
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp


def fixture(tmp_path):
    source=ForecastStore(tmp_path/'source');replay=ForecastStore(tmp_path/'replay')
    day='2026-10-01';cutoff=brussels_cutoff_timestamp(pd.Timestamp(day).date())
    row=pd.DataFrame(dict(run_date=day,zone='BE',target='solar',model='timestral_online',model_label='Timestral',
         source='historical_mock_replay',run_at='2026-11-10T00:00Z',timestamp=pd.date_range(cutoff,periods=30,freq='h'),forecast_mw=10.,actual_mw=12.))
    source.append_forecasts(row,day)
    live=row.assign(model='chronos2_online',source='online',forecast_mw=13.)
    source.append_forecasts(live,day)
    proof=dict(cutoff_utc=cutoff.isoformat(),snapshots=[dict(path='data/updates/a.csv',available_at=(cutoff-pd.Timedelta(hours=1)).isoformat(),sha256='x')])
    replay.append_forecasts(row.assign(forecast_mw=11.,actual_mw=999.,weather_vintage_policy='weather_collection_asof_v1',weather_vintage_provenance=json.dumps(proof)),day)
    audit=tmp_path/'audit.csv';row[['run_date','zone','target','model']].drop_duplicates().assign(status='weather_available').to_csv(audit,index=False)
    return source,replay,audit


def test_staging_preserves_live_predictions_and_existing_scoring_truth(tmp_path):
    source,replay,audit=fixture(tmp_path)
    result=stage(source.data_dir,replay.data_dir,audit,tmp_path/'staged')
    assert result['changed_groups']==1
    after=ForecastStore(tmp_path/'staged').read_forecasts()
    assert after[after.source.eq('online')].forecast_mw.eq(13).all()
    changed=after[after.source.eq('historical_mock_replay')]
    assert changed.forecast_mw.eq(11).all() and changed.actual_mw.eq(12).all()
    assert changed.weather_provenance_id.str.len().eq(64).all()
    assert source.read_forecasts()[lambda x:x.source.eq('historical_mock_replay')].forecast_mw.eq(10).all()


def test_staging_rejects_later_collection_before_creating_output(tmp_path):
    source,replay,audit=fixture(tmp_path)
    f=replay.read_forecasts();proof=json.loads(f.weather_vintage_provenance.iloc[0]);proof['snapshots'][0]['available_at']='2026-10-02T20:00Z'
    replay.append_forecasts(f.assign(weather_vintage_provenance=json.dumps(proof)),'2026-10-01')
    with pytest.raises(ValueError,match='later-collected'):
        stage(source.data_dir,replay.data_dir,audit,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_staging_cannot_replace_genuine_live_group(tmp_path):
    source,replay,audit=fixture(tmp_path)
    f=replay.read_forecasts();replay.append_forecasts(f.assign(model='chronos2_online'),'2026-10-01')
    with pytest.raises(ValueError,match='non-mock'):
        stage(source.data_dir,replay.data_dir,audit,tmp_path/'bad')
