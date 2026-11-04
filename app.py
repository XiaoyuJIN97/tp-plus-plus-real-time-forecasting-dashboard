"""Streamlit entry point. UI revision: 2026-11-03, dedicated probabilistic leaderboard."""
# Published dashboard data: fa619497b33b3911a21c45a1f5e0a99bbd6a99cd1c01f4fb00f0cc12093ca2f4

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_forecast_dashboard.ui.layout import render_app


render_app()
