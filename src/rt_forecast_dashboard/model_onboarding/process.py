"""Bounded, dependency-isolated inference shared by onboarding and production."""
from __future__ import annotations
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import threading

ROOT = Path(__file__).resolve().parents[3]


class ModelProcess:
    def __init__(self, python, factory, config, *, timeout=30, startup_timeout=180,
                 memory_mb=4096, threads=2, log_path=None):
        self.python, self.factory, self.config = str(python), factory, config
        self.timeout, self.startup_timeout, self.memory_mb = timeout, startup_timeout, memory_mb
        self.threads, self.log_path = threads, log_path
        self.process = None
        self.peak_rss_mb = 0.
        self.startup_seconds = 0.
        self.info = {}
        self._log = None
        self._lock = threading.Lock()

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            for handle in (process.stdin,process.stdout):
                try:
                    handle.close()
                except (BrokenPipeError,OSError):
                    pass
        if self._log is not None:
            self._log.close()
            self._log = None

    def _exchange(self, request, timeout):
        import psutil
        start = time.monotonic()
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
            self.process.stdin.flush()
            while True:
                process = psutil.Process(self.process.pid)
                family = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in family if p.is_running()) / 1024**2
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
                if rss > self.memory_mb:
                    raise MemoryError(f'Candidate exceeded {self.memory_mb} MiB RSS budget')
                if time.monotonic()-start > timeout:
                    raise TimeoutError(f'Candidate exceeded {timeout}s timeout')
                ready, _, _ = select.select([self.process.stdout], [], [], .05)
                if ready:
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError('Candidate worker exited without a response')
                    response = json.loads(line)
                    self.peak_rss_mb = max(self.peak_rss_mb, response.get('peak_rss_mb', 0))
                    if self.peak_rss_mb > self.memory_mb:
                        raise MemoryError('Candidate peak RSS exceeded budget')
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    return response
        except Exception:
            self.close()
            raise

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        allowed = {'PATH','HOME','TMPDIR','TEMP','TMP','LANG','LC_ALL','SYSTEMROOT',
                   'HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TORCH_HOME','XDG_CACHE_HOME',
                   'LD_LIBRARY_PATH','DYLD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES'}
        if self.process is not None:
            self.close()
        env = {**{k:v for k,v in os.environ.items() if k in allowed},
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'OMP_NUM_THREADS': str(self.threads), 'MKL_NUM_THREADS': str(self.threads),
               'OPENBLAS_NUM_THREADS': str(self.threads),
               'PYTHONPATH': str(ROOT/'src')+os.pathsep+str(ROOT)+os.pathsep+os.getenv('PYTHONPATH', '')}
        if self.log_path:
            self._log = open(self.log_path, 'a')
        self.process = subprocess.Popen([self.python, '-u', '-m', 'rt_forecast_dashboard.model_onboarding.worker'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                        stderr=self._log, env=env, start_new_session=True)
        start = time.monotonic()
        self.info = self._exchange({'initialize': True, 'factory': self.factory, 'config': self.config}, self.startup_timeout)
        self.startup_seconds = time.monotonic()-start

    def predict(self, request):
        with self._lock:
            return self._predict(request)

    def _predict(self, request):
        cold = self.process is None or self.process.poll() is not None
        started = time.monotonic()
        if cold:
            self.start()
        remaining = max(.001,self.startup_timeout-(time.monotonic()-started)) if cold else self.timeout
        response = self._exchange(request, remaining)
        return response['predictions'], cold
