"""Streamlit entry point. UI revision: 2026-11-03, dedicated probabilistic leaderboard."""
# Published dashboard data: 4b75b7372f362bf840cd8f40f3d2ab8b3fde897d33f25202e47f6a7799ee3242

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_forecast_dashboard.ui.layout import render_app


render_app()
