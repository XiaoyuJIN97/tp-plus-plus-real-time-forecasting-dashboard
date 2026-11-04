"""Update the watched Streamlit entry point when a dashboard snapshot changes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def update_revision(app: Path, dashboard: Path) -> str:
    digest = hashlib.sha256()
    # These are the inputs to actual lines and point-score views. A deterministic
    # digest avoids an app reload/commit when the refresh produced identical data.
    for name in ("actuals_recent.parquet", "forecasts_recent.parquet"):
        path = dashboard / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    revision = digest.hexdigest()
    source = app.read_text()
    source, count = re.subn(
        r"^# Published dashboard data: .*$",
        f"# Published dashboard data: {revision}",
        source,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Expected exactly one published-dashboard revision in app.py")
    app.write_text(source)
    return revision


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    update_revision(root / "app.py", root / "data/dashboard")
