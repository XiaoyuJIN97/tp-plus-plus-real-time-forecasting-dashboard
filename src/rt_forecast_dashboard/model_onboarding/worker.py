"""Candidate-only JSON-lines process. Never imports collectors or ForecastStore."""
from __future__ import annotations
import contextlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import socket
import sys

from .contracts import frames, prediction
from .environment import canonical


def main():
    def no_network(*args, **kwargs):
        raise RuntimeError('Network disabled: prepare archives and model cache before inference')
    # Defense against accidental requests; not a security boundary for untrusted code.
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    adapter = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get('initialize'):
                    module, name = request['factory'].split(':', 1)
                    adapter = getattr(importlib.import_module(module), name)(request['config'])
                    if 'torch' in sys.modules:
                        sys.modules['torch'].set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
                    distributions=list(importlib.metadata.distributions())
                    active,commits={},{}
                    for package in distributions:
                        name=package.metadata['Name']
                        if not name: continue
                        key=canonical(name)
                        if key in active: continue
                        active[key]=package.version
                        direct=package.read_text('direct_url.json')
                        if direct:
                            commits[key]=json.loads(direct).get('vcs_info',{}).get('commit_id')
                    response = {'active_versions':active,'source_revisions':commits,'ready': True, 'python': sys.version, 'platform': platform.platform(), 'system':platform.system(),
                                'processor':platform.processor(), 'cpu_count':os.cpu_count(),
                                'packages': sorted(f'{p.metadata["Name"]}=={p.version}' for p in distributions)}
                else:
                    if adapter is None:
                        raise RuntimeError('Worker not initialized')
                    past, future = frames(request['context'], request['future'], request['covariates'],
                                          context_hours=request['context_hours'], horizon=request['horizon'])
                    values = adapter.predict(future, request['target'], context=past, covariates=request['covariates'])
                    response = {'predictions': prediction(values, request['horizon']).tolist()}
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                response['peak_rss_mb'] = peak / (1024*1024 if sys.platform == 'darwin' else 1024)
        except Exception as exc:
            response = {'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
