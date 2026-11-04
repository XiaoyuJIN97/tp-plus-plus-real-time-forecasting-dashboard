from pathlib import Path

import pytest

from scripts.update_dashboard_revision import update_revision


def test_publication_reload_tracks_actuals_and_forecasts_without_noop_changes(tmp_path):
    app = tmp_path / 'app.py'
    app.write_text('"""App."""\n# Published dashboard data: initial\nprint("app")\n')
    dashboard = tmp_path / 'dashboard'
    dashboard.mkdir()
    (dashboard / 'actuals_recent.parquet').write_bytes(b'actuals v1')
    (dashboard / 'forecasts_recent.parquet').write_bytes(b'forecasts v1')
    first = update_revision(app, dashboard)
    source = app.read_text()
    assert update_revision(app, dashboard) == first
    assert app.read_text() == source
    (dashboard / 'actuals_recent.parquet').write_bytes(b'actuals v2')
    second = update_revision(app, dashboard)
    assert second != first
    (dashboard / 'forecasts_recent.parquet').write_bytes(b'forecasts v2')
    assert update_revision(app, dashboard) != second
    assert 'print("app")' in app.read_text()


def test_publication_refuses_missing_snapshot_or_missing_reload_marker(tmp_path):
    app = tmp_path / 'app.py'
    app.write_text('print("app")\n')
    with pytest.raises(FileNotFoundError):
        update_revision(app, tmp_path)
    for name in ('actuals_recent.parquet','forecasts_recent.parquet'):
        (tmp_path / name).write_bytes(b'data')
    with pytest.raises(ValueError, match='exactly one'):
        update_revision(app, tmp_path)
    assert app.read_text() == 'print("app")\n'
