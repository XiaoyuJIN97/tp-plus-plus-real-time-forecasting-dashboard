import pandas as pd
import pytest
from rt_forecast_dashboard.data.weather_vintages import WeatherVintages, WeatherVintageError


def snapshot(root, name, collected, values, phase='forecast', row_collected=None):
    path='data/updates/BE/solar/'+name+'.csv'
    dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'timestamp_utc':pd.date_range('2026-10-01T16:00Z',periods=4,freq='h'),
                  'updated_at_utc':row_collected or collected,'x':values}).to_csv(dest,index=False)
    manifest=root/'data/update_manifest.csv'
    row=pd.DataFrame([dict(country='BE',target='solar',path=path,collection_time_utc=collected,phase=phase)])
    if manifest.exists(): row=pd.concat([pd.read_csv(manifest),row],ignore_index=True)
    row.to_csv(manifest,index=False)


def test_excludes_later_versions_and_never_reads_annual_file(tmp_path):
    snapshot(tmp_path,'before','2026-10-01T17:59Z',[1,2,3,4])
    snapshot(tmp_path,'after','2026-10-01T18:01Z',[90]*4)
    past,future,proof=WeatherVintages(tmp_path).select('BE','solar',pd.Timestamp('2026-10-01T18:00Z'),2,2)
    assert past.x.tolist()==[1,2] and future.x.tolist()==[3,4]
    assert len(proof['snapshots'])==1 and proof['snapshots'][0]['sha256']


def test_row_timestamp_and_historical_phase_cannot_bypass_cutoff(tmp_path):
    snapshot(tmp_path,'historical','2026-10-01T17:00Z',[1]*4,phase='history')
    snapshot(tmp_path,'tampered','2026-10-01T17:30Z',[2]*4,row_collected='2026-10-02T00:00Z')
    with pytest.raises(WeatherVintageError,match='future horizon'):
        WeatherVintages(tmp_path).select('BE','solar',pd.Timestamp('2026-10-01T18:00Z'),2,2)


def test_missing_context_is_not_filled_with_later_snapshot(tmp_path):
    snapshot(tmp_path,'before','2026-10-01T17:00Z',[None,2,3,4])
    snapshot(tmp_path,'later','2026-10-02T17:00Z',[1,2,3,4])
    with pytest.raises(WeatherVintageError,match='1/2 context'):
        WeatherVintages(tmp_path).select('BE','solar',pd.Timestamp('2026-10-01T18:00Z'),2,2)


def test_latest_incomplete_future_falls_back_to_coherent_earlier_snapshot(tmp_path):
    snapshot(tmp_path,'before','2026-10-01T17:00Z',[1,2,3,4])
    snapshot(tmp_path,'newer','2026-10-01T17:30Z',[5,6,7,None])
    past,future,_=WeatherVintages(tmp_path).select('BE','solar',pd.Timestamp('2026-10-01T18:00Z'),2,2)
    assert future.x.tolist()==[3,4] and past.x.tolist()==[5,6]


def test_no_manifest_means_no_annual_fallback(tmp_path):
    with pytest.raises(WeatherVintageError,match='manifest'):
        WeatherVintages(tmp_path)


def test_replay_client_never_falls_back_to_live_api(tmp_path, monkeypatch):
    from rt_forecast_dashboard.data.weather_client import OpenMeteoClient
    from datetime import date
    monkeypatch.setenv('FORECAST_REPLAY_ASOF','true')
    monkeypatch.setenv('FORECAST_WEATHER_SNAPSHOT_ROOT',str(tmp_path))
    client=OpenMeteoClient(demo_mode=False)
    def forbidden(**kwargs):
        raise AssertionError('Replay requested later live weather')
    monkeypatch.setattr(client,'_fetch_openmeteo',forbidden)
    with pytest.raises(WeatherVintageError,match='manifest'):
        client.fetch_future_weather(run_date=date(2026,10,1),zone='BE',points=[],target='solar',horizon_hours=30)


def test_past_date_is_strict_even_without_replay_environment(tmp_path, monkeypatch):
    from rt_forecast_dashboard.data.weather_client import OpenMeteoClient
    from datetime import date
    monkeypatch.delenv('FORECAST_REPLAY_ASOF',raising=False)
    monkeypatch.setenv('FORECAST_WEATHER_SNAPSHOT_ROOT',str(tmp_path))
    client=OpenMeteoClient(demo_mode=False)
    with pytest.raises(WeatherVintageError,match='manifest'):
        client.fetch_future_weather(run_date=date(2020,1,1),zone='BE',points=[],target='solar',horizon_hours=30)
