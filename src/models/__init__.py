"""ESM-2 activation access and sparse dictionary learning."""

from models.esm_hooks import (
    ESMActivationExtractor,
    ESMWrapper,
    HookHandleSet,
    resolve_encoder_layers,
)
from models.sparse_autoencoder import SAEConfig, SAEOutput, TopKSparseAutoencoder

__all__ = [
    "ESMActivationExtractor",
    "ESMWrapper",
    "HookHandleSet",
    "SAEConfig",
    "SAEOutput",
    "TopKSparseAutoencoder",
    "resolve_encoder_layers",
]
