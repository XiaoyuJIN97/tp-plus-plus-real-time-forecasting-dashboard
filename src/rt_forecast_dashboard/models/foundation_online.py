"""Online adapters with a persistent, dependency-isolated inference process."""
import atexit
import json
import os
from pathlib import Path
import subprocess
import select
import threading

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
_WORKERS = {}
_LOCK = threading.Lock()


def close_workers():
    for process in _WORKERS.values():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    _WORKERS.clear()


atexit.register(close_workers)


class FoundationOnlineAdapter:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = os.getenv("FOUNDATION_DEVICE", device)

    def predict_batch(self, contexts, covariates, horizon=24):
        python = os.getenv("FOUNDATION_PYTHON", str(ROOT / ".foundation-env/bin/python"))
        cache = os.getenv("FOUNDATION_CACHE", str(Path.home() / ".cache/huggingface/hub"))
        key = (python, self.model, self.device, cache)
        with _LOCK:
            process = _WORKERS.get(key)
            if process is None or process.poll() is not None:
                env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.getenv("PYTHONPATH", "")}
                process = subprocess.Popen(
                    [python, "-u", "-m", "rt_forecast_dashboard.models.foundation_worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                )
                _WORKERS[key] = process
            request = dict(model=self.model, device=self.device, cache=cache,
                           compile=os.getenv("FOUNDATION_COMPILE", "false").lower() == "true",
                           contexts=np.asarray(contexts).tolist(),
                           covariates=np.asarray(covariates).tolist(), horizon=horizon)
            process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], float(os.getenv("FOUNDATION_TIMEOUT_SECONDS", "900")))
            if not ready:
                process.kill()
                process.wait()
                raise TimeoutError(f"{self.model} inference timed out")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.model} inference worker exited ({process.poll()})")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(response["error"])
        result = np.asarray(response["predictions"], dtype=float)
        if result.shape != (len(contexts), horizon) or not np.isfinite(result).all():
            raise ValueError("Invalid foundation forecast shape or values")
        return np.maximum(result, 0)

    def predict(self, features, target, context=None, covariates=None):
        if context is None or len(context) < 24 * 7:
            raise ValueError("Foundation inference requires at least seven days of actual context")
        context = context.sort_values("timestamp").copy()
        future = features.sort_values("timestamp").copy()
        cols = list(covariates or ["tso_forecast_mw"])
        if any(col not in context or col not in future for col in cols):
            raise ValueError("Required foundation covariates are missing")
        for frame in (context, future):
            frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()
        x = context.actual_mw.to_numpy(dtype=np.float32)
        cov = pd.concat([context[cols], future[cols]]).to_numpy(dtype=np.float32).T
        if not np.isfinite(x).all() or not np.isfinite(cov).all():
            raise ValueError("Nonfinite foundation context or covariates")
        if pd.to_datetime(context.timestamp, utc=True).max() >= pd.to_datetime(future.timestamp, utc=True).min():
            raise ValueError("Actual context overlaps forecast horizon")
        result = self.predict_batch([x], [cov], len(future))[0]
        return pd.Series(result, index=future.index).reindex(features.index).to_numpy()
