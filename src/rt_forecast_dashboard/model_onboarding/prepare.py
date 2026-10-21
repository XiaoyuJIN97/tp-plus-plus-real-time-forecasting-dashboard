"""Download a permitted checkpoint from its publisher before network-free inference."""
import json
import os
from pathlib import Path
import subprocess
import time
import yaml
from .process import ROOT
from .contracts import staging_path


def prepare(manifest_path, python, output):
    output=staging_path(output)
    spec=yaml.safe_load(Path(manifest_path).read_text())
    module,name=spec['prepare_factory'].split(':',1)
    allowed={'PATH','HOME','TMPDIR','TEMP','LANG','HF_HOME','HF_HUB_CACHE','TORCH_HOME','XDG_CACHE_HOME'}
    env={k:v for k,v in os.environ.items() if k in allowed}
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    command='import importlib,json,sys; getattr(importlib.import_module(sys.argv[1]),sys.argv[2])(json.loads(sys.argv[3]))'
    start=time.monotonic()
    result=subprocess.run([str(python),'-c',command,module,name,json.dumps(spec['adapter_config'])],
                          env=env,capture_output=True,text=True,timeout=900)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps({'seconds':time.monotonic()-start,'returncode':result.returncode,
                                       'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
    if result.returncode: raise RuntimeError('Checkpoint preparation failed; see preparation log')
