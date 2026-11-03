"""Guard matched-window comparisons and the historical grid lineage."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.build_solar_forecast_package import complete_panel, metrics, PACKAGE
from rt_forecast_dashboard.time_utils import forecast_delivery_grid


def test_panel_requires_full_shared_grid_and_uses_one_truth():
    grid=forecast_delivery_grid('2026-10-24')
    rows=[]
    for model in ('tso_reference','persistence'):
        for stamp in grid:
            rows.append(dict(zone='BE',run_date='2026-10-24',timestamp=stamp,model=model,
                             forecast_mw=10.,actual_mw=9. if model=='tso_reference' else 500.))
    frame=pd.DataFrame(rows)
    panel=complete_panel(frame,['tso_reference','persistence'])
    assert len(panel)==31 and panel.truth.eq(9.).all()
    missing=frame.drop(frame[frame.model.eq('persistence')].index[-1])
    assert complete_panel(missing,['tso_reference','persistence']).empty
    frame.timestamp+=pd.Timedelta(hours=1)
    assert complete_panel(frame,['tso_reference','persistence']).empty


def test_wape_includes_zero_generation_hours_without_dividing_by_zero():
    frame=pd.DataFrame(dict(run_date=['2026-10-01']*2,
        timestamp=pd.to_datetime(['2026-10-01T16:00Z','2026-10-01T17:00Z']),
        truth=[0.,100.],tso_reference=[10.,90.],persistence=[0.,50.]))
    result=metrics(frame,'tso_reference')
    assert result['wape_pct']==20.
    assert result['mae_mw']==10.
    assert np.isfinite(result['rmse_mw'])


def test_extracted_original_snapping_matches_recorded_centroids():
    spec=importlib.util.spec_from_file_location('original_selection',PACKAGE/'original_point_selection.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metadata=pd.read_csv(PACKAGE/'original_selection_metadata.csv')
    used=set()
    for row in metadata.itertuples():
        lat,lon=module.nearest_unused_era5_grid_point(row.reference_lat,row.reference_lon,used)
        used.add((round(lat,3),round(lon,3)))
        assert (lat,lon)==(row.era5_lat,row.era5_lon)
        glat,glon=module.nearest_gfs_grid_point(lat,lon)
        assert (round(glat,6),round(glon,6))==(row.gfs_lat,row.gfs_lon)
