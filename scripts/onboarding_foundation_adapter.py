"""Worked CPU adapters for the current T0 Gamma and Timestral checkpoints.

Used inside the onboarding worker, never the dashboard interpreter. No collector,
ForecastStore or nested inference subprocess. Input validation is fail-closed.
"""
from pathlib import Path
import numpy as np
from rt_forecast_dashboard.model_onboarding.contracts import frames, prediction

CHECKPOINTS = {
    't0_gamma': ('theforecastingcompany/t0-gamma', 'e08aee083778f1f32da4974c0a34276776a35c52'),
    'timestral': ('mistralai/Timestral-209M', '9874d252ab0aae16f7f7c999f98f98d6ad7d8365'),
}


def _checkpoint(config):
    expected = CHECKPOINTS[config['family']]
    if (config['model_id'], config['revision']) != expected or config['device'] != 'cpu':
        raise ValueError('Example supports only the reviewed pinned CPU checkpoints')
    return expected


def prepare_model(config):
    from huggingface_hub import snapshot_download
    model_id, revision = _checkpoint(config)
    snapshot_download(model_id, revision=revision, local_dir=config['snapshot_dir'])


class Adapter:
    def __init__(self, config):
        _checkpoint(config)
        snapshot = Path(config['snapshot_dir']).resolve()
        if not (snapshot/'config.json').is_file():
            raise ValueError('Run prepare first; a complete local snapshot is required')
        self.family = config['family']
        # Local paths plus the worker's offline/network guards prevent downloads.
        if self.family == 't0_gamma':
            from t0 import T0Forecaster
            self.model = T0Forecaster.from_pretrained(str(snapshot)).eval().to('cpu')
        else:
            from timestral import TimestralForecaster
            self.model = TimestralForecaster.from_pretrained(str(snapshot), device='cpu')

    def predict(self, features, target, context=None, covariates=None):
        import torch
        cols = list(covariates or [])
        past, future = frames(context, features, cols)
        # Shared arrays: B=1, L=2208, H=24, C=len(cols).
        x = past.actual_mw.to_numpy(np.float32)[None, :]
        cov = np.concatenate([past[cols].to_numpy(np.float32),
                              future[cols].to_numpy(np.float32)]).T[None, :, :]
        with torch.inference_mode():
            if self.family == 't0_gamma':
                output = self.model.predict(x, horizon=24, future_covariates=cov,
                    quantile_levels=[0.5], trim_leading_missing=True,
                    symmetric_averaging=True, clip_nonnegative=True,
                    snap_integers=True).median
            else:
                output = self.model.forecast(x[:, None, :], prediction_length=24,
                                            known_covariates=cov).median[:, 0, :]
        if hasattr(output, 'detach'):
            output = output.detach().cpu().numpy()
        result = np.asarray(output)
        if result.shape != (1, 24) or not np.isfinite(result).all():
            raise ValueError(f'Unexpected native output: {result.shape}')
        # Match current foundation_runtime.py's explicit nonnegative MW policy.
        return prediction(np.maximum(result[0], 0), 24)


def create_adapter(config):
    return Adapter(config)
