"""Device selection, with the Apple Silicon caveats spelled out.

MPS is the point of running this locally, but it has two behaviours that bite in
an interpretability codebase specifically:

1. Unimplemented kernels raise instead of falling back. Several ops used in
   attribution (notably some autograd paths and ``index_put_`` variants) are
   still missing in older torch builds. ``PYTORCH_ENABLE_MPS_FALLBACK=1`` routes
   those to the CPU instead of crashing the run 40 minutes in. It must be set
   *before* torch is imported, which is why :func:`enable_mps_fallback` writes
   the environment variable and warns if torch is already loaded.

2. float64 is not supported at all. Anything that wants double precision — the
   permutation null in the PDB validator, for instance — must run on CPU or in
   float32. Helpers here keep that explicit rather than letting it surface as a
   cryptic dtype error.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import torch

__all__ = [
    "autocast_dtype",
    "device_report",
    "enable_mps_fallback",
    "resolve_device",
    "to_device",
]


def enable_mps_fallback() -> None:
    """Allow unimplemented MPS ops to fall back to CPU.

    Safe to call more than once. Call it before importing torch for it to take
    full effect; if torch is already imported we still set the variable (it is
    read lazily by some code paths) but warn that it may be too late.
    """
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        return
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if "torch" in sys.modules:
        warnings.warn(
            "PYTORCH_ENABLE_MPS_FALLBACK was set after torch was imported; "
            "set it at the top of your entry point to be certain it applies.",
            RuntimeWarning,
            stacklevel=2,
        )


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Turn a config string into a concrete :class:`torch.device`.

    ``"auto"`` prefers MPS, then CUDA, then CPU. An explicit request for a
    device that is not available raises rather than silently downgrading — a
    silent downgrade to CPU on a 12-hour sweep is the kind of thing you only
    notice the next morning.
    """
    if isinstance(spec, torch.device):
        return spec
    if spec is None or spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    spec = str(spec)
    if spec.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError(
            "device='mps' requested but MPS is unavailable. On Apple Silicon this "
            "usually means a non-arm64 torch build; reinstall with "
            "`pip install --force-reinstall torch`."
        )
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable.")
    return torch.device(spec)


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Preferred reduced-precision dtype for the given device.

    MPS supports float16 but not bfloat16 on most builds; CUDA prefers bfloat16;
    CPU autocast is rarely worth it, so we stay in float32 there.
    """
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def supports_float64(device: torch.device) -> bool:
    """MPS has no float64. Anything needing doubles must move to CPU."""
    return device.type != "mps"


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors inside dicts / lists / tuples to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


def device_report(device: torch.device) -> str:
    """One-line human-readable summary, logged at the start of every script."""
    parts = [f"device={device}"]
    if device.type == "mps":
        parts.append(f"mps_fallback={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')}")
        parts.append("float64=unsupported")
    elif device.type == "cuda":
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        parts.append(f"vram={total:.1f}GiB")
    parts.append(f"torch={torch.__version__}")
    return "  ".join(parts)


def empty_cache(device: torch.device) -> None:
    """Release cached blocks. Matters on an 8 GB machine between sweep stages."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
