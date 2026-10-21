"""Runnable protocol example, NOT a TSFM or an admitted production model."""
import numpy as np


class DemoAdapter:
    def predict(self, features, target, context=None, covariates=None):
        # Replace only this vendor-specific mapping with actual model inference.
        # context.actual_mw is past-only; features contains known future covariates.
        return np.maximum(context.actual_mw.to_numpy(float)[-len(features):], 0)


def create_adapter(config):
    # A real factory loads the pinned checkpoint locally, with networking disabled.
    return DemoAdapter()


def prepare_model(config):
    # Demo has no weights. A real adapter can use, for example:
    # snapshot_download(repo_id=config['model_id'], revision=config['revision'])
    # Only download from the publisher after checking terms; do not upload weights.
    pass
