"""Reproducibility helpers.

Full determinism is not achievable on MPS — several kernels are nondeterministic
and there is no equivalent of ``torch.use_deterministic_algorithms`` coverage
there. What we can do is seed every generator we touch and record the seed with
the results, so a run is reproducible up to kernel-level float nondeterminism.
:func:`seed_everything` returns the seed so callers can log it.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "torch_generator"]


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUDA-only; on MPS these are no-ops and cuDNN flags do not exist.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def torch_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """A seeded generator for sampling that must not disturb global RNG state.

    Used by the permutation null in the PDB validator and by the activation
    shuffle buffer, both of which should be reproducible independently of how
    many batches the training loop happened to draw first.
    """
    device = torch.device(device)
    # Generators on MPS are not supported in all torch versions; CPU is fine
    # since these draws are small and then moved.
    gen_device = "cpu" if device.type == "mps" else device
    g = torch.Generator(device=gen_device)
    g.manual_seed(seed)
    return g
