"""Production adapter for independently installed, admitted candidate models."""
import atexit
import json
from pathlib import Path
from rt_forecast_dashboard.model_onboarding.contracts import frames, prediction, records
from rt_forecast_dashboard.model_onboarding.process import ModelProcess, ROOT


_WORKERS = {}


class IsolatedAdapter:
    def __init__(self, config):
        # Lazy setup: errors stay inside run_daily's per-model exception boundary.
        self.config = config
        self.worker = None

    def predict(self, features, target, context=None, covariates=None):
        cols = list(covariates or [])
        past, future = frames(context, features, cols)
        if self.worker is None:
            cfg = self.config
            python = Path(cfg['python'])
            if not python.is_absolute():
                python = ROOT/python
            key = json.dumps({**cfg, 'python':str(python)}, sort_keys=True)
            if key in _WORKERS:
                self.worker = _WORKERS[key]
            else:
                self.worker = ModelProcess(python, cfg['adapter_factory'], cfg.get('adapter_config', {}),
                                       timeout=cfg.get('timeout_seconds', 30),
                                       startup_timeout=cfg.get('startup_timeout_seconds', 180),
                                       memory_mb=cfg.get('memory_mb', 4096), threads=cfg.get('threads', 2))
                _WORKERS[key] = self.worker
                atexit.register(self.worker.close)
        values, _ = self.worker.predict(dict(context=records(past), future=records(future),
                                             covariates=cols, target=target, context_hours=len(past), horizon=len(future)))
        return prediction(values, len(future))
