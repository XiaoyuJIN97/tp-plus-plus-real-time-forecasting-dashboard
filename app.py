"""Streamlit entry point. UI revision: 2026-11-03, dedicated probabilistic leaderboard."""
# Published dashboard data: eb49217ce20fece0f40008497e6a6b079e259770e3a88557f15bc52f8b47c6b5

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_forecast_dashboard.ui.layout import render_app


render_app()
