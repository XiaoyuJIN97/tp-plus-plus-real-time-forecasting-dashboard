from rt_forecast_dashboard.config import zones


def test_dashboard_uses_supported_zones_and_excludes_netherlands() -> None:
    configured = zones()

    assert set(configured) == {"BE", "FR", "DE", "DK1", "DK2", "ES", "PT"}
    assert configured["ES"]["targets"] == ["load", "solar", "wind_onshore"]
    assert all("weather_points" in zone for zone in configured.values())
