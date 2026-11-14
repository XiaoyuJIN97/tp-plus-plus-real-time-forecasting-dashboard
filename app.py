"""Streamlit entry point. UI revision: 2026-11-03, dedicated probabilistic leaderboard."""
# Published dashboard data: 2342de3693cd8d3b56ebf1d0756e58849f25a2e96b82ec4f7cf6cf74e400658b

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_forecast_dashboard.ui.layout import render_app


render_app()
