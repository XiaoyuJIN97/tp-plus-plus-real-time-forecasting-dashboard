"""Pinned native inference shared by offline evaluation and isolated online workers."""
from pathlib import Path
import os
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
REVISIONS = {
    't0_gamma': 'e08aee083778f1f32da4974c0a34276776a35c52',
    'timestral': '9874d252ab0aae16f7f7c999f98f98d6ad7d8365',
}

class FoundationRuntime:
    def __init__(self, name, device, cache, compile_model=False):
        import torch
        self.name = name
        if name == 't0_gamma':
            from t0 import T0Forecaster
            self.model = T0Forecaster.from_pretrained('theforecastingcompany/t0-gamma', revision=REVISIONS[name], cache_dir=str(cache)).eval().to(device)
        else:
            from timestral import TimestralForecaster
            if device == 'mps':
                # Official scaling accumulates in float64, unavailable on MPS.
                # Keep that exact calculation on CPU; neural inference stays on GPU.
                import timestral.scaler as scaler
                original = scaler.compute_causal_statistics
                if not getattr(original, '_cpu_for_mps', False):
                    def statistics(data, mask, *args, **kwargs):
                        if data.device.type != 'mps':
                            return original(data, mask, *args, **kwargs)
                        result = original(data.cpu(), mask.cpu(), *args, **kwargs)
                        return tuple(value.to(data.device) for value in result)
                    statistics._cpu_for_mps = True
                    scaler.compute_causal_statistics = statistics
            self.model = TimestralForecaster.from_pretrained('mistralai/Timestral-209M', revision=REVISIONS[name], device=device, cache_dir=str(cache))
        if compile_model and device == 'mps' and name == 'timestral':
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT/'.codex-build/torchinductor'))
            self.model.model._run_two_stream = torch.compile(self.model.model._run_two_stream, dynamic=False)

    def predict(self, contexts, covariates, horizon=24):
        import torch
        x = np.stack(contexts).astype(np.float32)
        cov = np.stack(covariates).astype(np.float32)
        with torch.inference_mode():
            if self.name == 't0_gamma':
                result = self.model.predict(x, horizon=horizon, future_covariates=cov,
                    quantile_levels=[0.5], trim_leading_missing=True,
                    symmetric_averaging=True, clip_nonnegative=True, snap_integers=True).median
            else:
                result = self.model.forecast(x[:, None, :], prediction_length=horizon, known_covariates=cov).median[:, 0, :]
        if hasattr(result, 'detach'):
            result = result.detach().cpu().numpy()
        result = np.asarray(result).reshape(len(x), horizon)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite model forecasts')
        return np.maximum(result, 0)
