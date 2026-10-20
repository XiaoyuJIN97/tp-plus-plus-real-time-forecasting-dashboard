"""JSON-lines inference service, launched inside the isolated model environment."""
import contextlib
import json
import os
import sys
from pathlib import Path

from rt_forecast_dashboard.models.foundation_runtime import FoundationRuntime


def main():
    import torch
    torch.set_num_threads(int(os.getenv("FOUNDATION_THREADS", "4")))
    model = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            # Model libraries may print startup messages; stdout is the protocol.
            with contextlib.redirect_stdout(sys.stderr):
                if model is None:
                    model = FoundationRuntime(
                        request["model"], request["device"],
                        Path(request["cache"]), request.get("compile", False))
                result = model.predict(request["contexts"], request["covariates"], request["horizon"])
            response = {"predictions": result.tolist()}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
