"""Model execution tracer that records per-module shapes, dtypes & timing.

Usage::

    model = LlamaModel(...)
    x = rand(...)

    with ModelTracer() as tracer:
        out = model(x)

    tracer.dump("model_trace.json")   # Chrome Trace format
"""

import gc
import json
import time
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


# ── helpers ──────────────────────────────────────────────────────────

def _assign_trace_names() -> None:
    """Walk all ``nn.Module`` objects in memory and set ``_trace_name``."""
    _CONTAINER_TYPES = (nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)
    seen: set[int] = set()
    for obj in gc.get_objects(0):
        try:
            if isinstance(obj, nn.Module) and id(obj) not in seen:
                seen.add(id(obj))
                if isinstance(obj, _CONTAINER_TYPES):
                    continue
                _name_children(obj, getattr(obj, "_trace_name", type(obj).__name__))
        except Exception:
            pass


def _name_children(module: nn.Module, prefix: str) -> None:
    for name, child in module.named_children():
        child._trace_name = f"{prefix}.{name}"
        _name_children(child, child._trace_name)


def _first_tensor(args: tuple, kwargs: dict[str, Any]) -> torch.Tensor | None:
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _io_str(t: torch.Tensor | None) -> tuple[str, str]:
    if t is None:
        return "N/A", "N/A"
    return str(tuple(t.shape)), str(t.dtype)


def _describe_output(output: Any) -> tuple[str, str]:
    if isinstance(output, torch.Tensor):
        return _io_str(output)
    if isinstance(output, (tuple, list)):
        shapes, dtypes = [], []
        for o in output:
            s, d = _describe_output(o)
            if s != "N/A":
                shapes.append(s)
                dtypes.append(d)
        return " | ".join(shapes) if shapes else "N/A", " | ".join(dtypes) if dtypes else "N/A"
    if isinstance(output, dict):
        for v in output.values():
            s, d = _describe_output(v)
            if s != "N/A":
                return s, d
    return "N/A", "N/A"


# ── tracer ───────────────────────────────────────────────────────────

class ModelTracer:
    """Context manager that monkey-patches ``nn.Module.__call__`` to trace
    every sub-module invocation inside the ``with`` block.

    Attributes
    ----------
    events : list[dict]
        Accumulated Chrome Trace events (one ``X`` complete event per call).
    """

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._call_stack: list[dict] = []
        self._orig_call: Any = None

    # ── internal hooks ───────────────────────────────────────────────

    def _pre(self, module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        name = type(module).__name__
        ts = time.perf_counter_ns() / 1000.0
        t = _first_tensor(args, kwargs)
        in_shape, in_dtype = _io_str(t)
        self._call_stack.append(dict(
            name=name, ts=ts,
            in_shape=in_shape, in_dtype=in_dtype,
            module_path=getattr(module, "_trace_name", name),
        ))

    def _post(self, output: Any) -> None:
        info = self._call_stack.pop()
        ts_end = time.perf_counter_ns() / 1000.0
        dur = ts_end - info["ts"]
        out_shape, out_dtype = _describe_output(output)
        self.events.append(dict(
            name=info["name"],
            ph="X",
            ts=round(info["ts"], 3),
            dur=round(dur, 3),
            pid=0,
            tid=0,
            cat=info["module_path"].rpartition(".")[0] or "root",
            args=dict(
                input_shape=info["in_shape"],
                input_dtype=info["in_dtype"],
                output_shape=out_shape,
                output_dtype=out_dtype,
                module_path=info["module_path"],
            ),
        ))

    

    # ── context manager ──────────────────────────────────────────────

    def __enter__(self) -> "ModelTracer":
        self.events.clear()
        self._call_stack.clear()

        # Pre-compute qualified names for every Module already alive.
        _assign_trace_names()

        self._orig_call = nn.Module.__call__
        tracer = self

        def traced_call(self_module: nn.Module, *args: Any, **kwargs: Any) -> Any:
            if isinstance(self_module, (nn.ModuleList, nn.ModuleDict)):
                return tracer._orig_call(self_module, *args, **kwargs)
            tracer._pre(self_module, args, kwargs)
            result: Any = None
            exc = True
            try:
                result = tracer._orig_call(self_module, *args, **kwargs)
                exc = False
                return result
            finally:
                if exc:
                    if tracer._call_stack:
                        tracer._call_stack.pop()
                else:
                    tracer._post(result)

        nn.Module.__call__ = traced_call  # type: ignore[method-assign]
        return self

    def __exit__(self, *args: Any) -> None:
        nn.Module.__call__ = self._orig_call  # type: ignore[method-assign]

    # ── serialisation ────────────────────────────────────────────────

    def dump(self, path: str) -> None:
        """Write events to *path* as a Chrome Trace JSON file."""
        if not self.events:
            print("[ModelTracer] no events recorded – was the model called inside the with block?")
        else:
            pass
        trace = OrderedDict([
            ("traceEvents", self.events),
            ("displayTimeUnit", "ms"),
        ])
        with open(path, "w") as f:
            json.dump(trace, f, indent=2)
