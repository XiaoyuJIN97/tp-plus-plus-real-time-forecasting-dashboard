"""Streamlit entry point. UI revision: 2026-11-03, dedicated probabilistic leaderboard."""
# Published dashboard data: c28fb135319bbe498e6d722e61b002a587a8487db944ba7275fdf66c7548f6a5

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_forecast_dashboard.ui.layout import render_app


render_app()
