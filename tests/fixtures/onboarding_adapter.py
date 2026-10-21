import os
import socket
import time
import numpy as np


class Adapter:
    def __init__(self, config):
        self.config, self.calls = config, 0

    def predict(self, features, target, context=None, covariates=None):
        self.calls += 1
        assert 'actual_mw' not in features
        assert os.getenv('ENTSOE_API_KEY') is None
        if self.config.get('behavior')=='network':
            socket.create_connection(('example.invalid',443))
        if self.config.get('behavior')=='hang' and self.calls>1:
            time.sleep(30)
        if self.config.get('behavior')=='crash':
            os._exit(7)
        if self.config.get('behavior')=='bad_shape':
            return np.zeros(2)
        values = context.actual_mw.to_numpy(float)[-len(features):].copy()
        context['actual_mw'] = -999
        features['mutated'] = True
        return values


def create_adapter(config):
    return Adapter(config)
