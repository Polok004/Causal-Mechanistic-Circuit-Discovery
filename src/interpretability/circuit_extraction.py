"""Assembling individual interventions into a circuit.

A circuit here is a small set of ``(layer, feature, positions)`` nodes whose
joint restoration recovers most of the model's wild-type behaviour on a given
mutation, plus the edges between nodes at different layers.

Three-stage discovery
---------------------
Scoring every feature exactly is a forward pass per feature per layer per
variant, which does not finish. The pipeline is therefore:

1. **Prune.** Keep only features active in the clean or corrupted run. With
   k=32 and a 5120-element dictionary this is already a ~50x reduction, and it
   is exact — an inactive feature has identical values in both runs, so patching
   it provably does nothing.
2. **Rank.** Score the survivors with attribution patching (one backward pass).
   Approximate, used only for ordering.
3. **Confirm.** Re-score the top candidates with exact path patching and keep
   those that survive. Everything reported downstream comes from this stage.

Greedy selection, and why it is not just "top-n"
-------------------------------------------------
Individually-high-DCE features are often redundant: several features encode the
same thing and each alone recovers most of the effect, so taking the top 5 by
individual DCE gives a "circuit" of five copies of one mechanism. Greedy
selection instead adds, at each step, the feature that most improves the
recovery of the set *already chosen*, which stops once additional features stop
contributing. That is the difference between a circuit and a leaderboard.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from interpretability.path_patching import (
    CausalPatcher,
    PatchingSetup,
    attribution_scores,
    positionwise_effect,
)
from utils.protein import token_to_seq_pos

__all__ = [
    "Circuit",
    "CircuitEdge",
    "CircuitNode",
    "discover_circuit",
    "greedy_select",
]


@dataclass
class CircuitNode:
    """One SAE feature at one layer, with the positions where it acts."""

    layer: int
    feature: int
    dce: float
    normalized_dce: float
    # Token positions whose individual DCE cleared the threshold.
    token_positions: list[int] = field(default_factory=list)
    position_effects: list[float] = field(default_factory=list)

    @property
    def residues(self) -> list[int]:
        """1-indexed residue numbers, the convention PDB files and papers use.

        Token positions are converted through the shared helper rather than by
        subtracting one inline, so the ``<cls>`` offset is handled in exactly one
        place in the codebase.
        """
        return [token_to_seq_pos(p) + 1 for p in self.token_positions]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["residues"] = self.residues
        return d


@dataclass
class CircuitEdge:
    """A causal link from an upstream node to a downstream node.

    ``weight`` is the change in the downstream feature's activation caused by
    restoring the upstream feature — a direct measurement, not a correlation
    between activations.
    """

    src_layer: int
    src_feature: int
    dst_layer: int
    dst_feature: int
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Circuit:
    """A discovered circuit plus everything needed to reproduce it."""

    nodes: list[CircuitNode] = field(default_factory=list)
    edges: list[CircuitEdge] = field(default_factory=list)

    # Provenance.
    model_name: str = "unknown"
    sequence: str = ""
    mutation: str = ""
    pdb_id: str | None = None
    clean_metric: float = 0.0
    corrupted_metric: float = 0.0
    joint_metric: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def recovered_fraction(self) -> float:
        """Fraction of the mutation's effect recovered by the full node set.

        This is the headline number for a circuit: patching *all* selected
        features together should recover most of the gap. A circuit whose parts
        each score well but whose joint recovery is low is not a circuit.
        """
        gap = self.clean_metric - self.corrupted_metric
        if abs(gap) < 1e-8:
            return 0.0
        return (self.joint_metric - self.corrupted_metric) / gap

    @property
    def residues(self) -> list[int]:
        """Sorted union of 1-indexed residues touched by any node."""
        out: set[int] = set()
        for node in self.nodes:
            out.update(node.residues)
        return sorted(out)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sequence": self.sequence,
            "mutation": self.mutation,
            "pdb_id": self.pdb_id,
            "clean_metric": self.clean_metric,
            "corrupted_metric": self.corrupted_metric,
            "joint_metric": self.joint_metric,
            "recovered_fraction": self.recovered_fraction,
            "residues": self.residues,
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "created_at": self.created_at,
            "config": self.config,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Circuit:
        raw = json.loads(Path(path).read_text())
        nodes = [
            CircuitNode(
                layer=n["layer"],
                feature=n["feature"],
                dce=n["dce"],
                normalized_dce=n["normalized_dce"],
                token_positions=n.get("token_positions", []),
                position_effects=n.get("position_effects", []),
            )
            for n in raw.get("nodes", [])
        ]
        edges = [CircuitEdge(**e) for e in raw.get("edges", [])]
        return cls(
            nodes=nodes,
            edges=edges,
            model_name=raw.get("model_name", "unknown"),
            sequence=raw.get("sequence", ""),
            mutation=raw.get("mutation", ""),
            pdb_id=raw.get("pdb_id"),
            clean_metric=raw.get("clean_metric", 0.0),
            corrupted_metric=raw.get("corrupted_metric", 0.0),
            joint_metric=raw.get("joint_metric", 0.0),
            created_at=raw.get("created_at", ""),
            config=raw.get("config", {}),
        )


# --------------------------------------------------------------------------- #
# Selection                                                                     #
# --------------------------------------------------------------------------- #


def greedy_select(
    patcher: CausalPatcher,
    setup: PatchingSetup,
    candidates: Sequence[int],
    *,
    max_features: int = 8,
    min_gain: float = 0.01,
    cache: dict[str, Any] | None = None,
    progress: bool = True,
) -> tuple[list[int], list[float]]:
    """Greedily grow a feature set by joint recovery.

    Args:
        candidates: features to choose from, ideally already shortlisted.
        max_features: hard cap on circuit size. A circuit that needs 50 features
            is not an explanation of anything.
        min_gain: stop when the best remaining feature improves normalised
            recovery by less than this. This is what makes the size adaptive
            rather than fixed.

    Returns:
        ``(chosen_features, cumulative_recovery)`` where the second list gives
        the joint normalised recovery after each addition.
    """
    cache = patcher.baselines(setup) if cache is None else cache
    gap = cache["clean_metric"] - cache["corrupted_metric"]
    if abs(gap) < 1e-8:
        return [], []

    remaining = list(candidates)
    chosen: list[int] = []
    trajectory: list[float] = []
    current = 0.0

    bar = tqdm(range(min(max_features, len(remaining))), disable=not progress, desc="greedy")
    for _ in bar:
        best_feature, best_recovery = None, current
        for f in remaining:
            res = patcher.direct_causal_effect(setup, chosen + [f], cache=cache)
            recovery = res.normalized_dce
            if recovery > best_recovery:
                best_feature, best_recovery = f, recovery

        if best_feature is None or (best_recovery - current) < min_gain:
            break
        chosen.append(best_feature)
        remaining.remove(best_feature)
        current = best_recovery
        trajectory.append(current)
        bar.set_postfix(recovery=f"{current:.3f}", n=len(chosen))

    return chosen, trajectory


def discover_edges(
    patcher_by_layer: dict[int, CausalPatcher],
    setup: PatchingSetup,
    nodes: Sequence[CircuitNode],
    *,
    threshold: float = 0.05,
    progress: bool = True,
) -> list[CircuitEdge]:
    """Measure how upstream nodes change downstream features' activations.

    For each upstream node, we patch it in the corrupted run and read the
    downstream SAE latents under that intervention. The edge weight is the
    change in the downstream feature's mean activation, normalised by its clean
    activation so weights are comparable across features of different scales.

    Only forward edges (lower layer -> higher layer) are considered: information
    in a transformer's residual stream flows one way, so a "backward edge" would
    be an artefact.
    """
    edges: list[CircuitEdge] = []
    ordered = sorted(nodes, key=lambda n: n.layer)

    pairs = [
        (src, dst)
        for i, src in enumerate(ordered)
        for dst in ordered[i + 1 :]
        if dst.layer > src.layer
    ]
    for src, dst in tqdm(pairs, disable=not progress, desc="edges"):
        src_patcher = patcher_by_layer.get(src.layer)
        dst_patcher = patcher_by_layer.get(dst.layer)
        if src_patcher is None or dst_patcher is None:
            continue

        with torch.no_grad():
            _, clean_dst = dst_patcher.sae_latents(setup.clean)
            _, corrupt_dst = dst_patcher.sae_latents(setup.corrupted)
            clean_cache = src_patcher.baselines(setup)

            with src_patcher._patched(clean_cache["clean_latents"], src.feature, None):
                _, patched_dst = dst_patcher.sae_latents(setup.corrupted)

        baseline = float(corrupt_dst[..., dst.feature].mean())
        patched = float(patched_dst[..., dst.feature].mean())
        reference = float(clean_dst[..., dst.feature].mean())
        denom = abs(reference - baseline)
        if denom < 1e-8:
            continue
        weight = (patched - baseline) / denom
        if abs(weight) >= threshold:
            edges.append(
                CircuitEdge(
                    src_layer=src.layer,
                    src_feature=src.feature,
                    dst_layer=dst.layer,
                    dst_feature=dst.feature,
                    weight=weight,
                )
            )
    return edges


# --------------------------------------------------------------------------- #
# Top-level driver                                                              #
# --------------------------------------------------------------------------- #


def discover_circuit(
    patchers: dict[int, CausalPatcher],
    setup: PatchingSetup,
    *,
    top_k_attribution: int = 64,
    max_features_per_layer: int = 4,
    min_gain: float = 0.01,
    position_threshold: float = 0.1,
    edge_threshold: float = 0.05,
    find_edges: bool = True,
    progress: bool = True,
) -> Circuit:
    """Run prune -> rank -> confirm -> localise -> link, for one variant.

    Args:
        patchers: one :class:`CausalPatcher` per layer to search.
        setup: the clean/corrupted pair.
        top_k_attribution: how many attribution-ranked candidates per layer go
            forward to exact patching.
        max_features_per_layer: cap on greedily-selected features per layer.
        position_threshold: a position is kept for a node when its individual
            DCE is at least this fraction of the node's largest positional
            effect. Controls how tight the residue set is.
        find_edges: edge discovery costs a forward pass per node pair; skip it
            when you only need the residues.
    """
    nodes: list[CircuitNode] = []
    selected_by_layer: dict[int, list[int]] = {}

    layer_iter = sorted(patchers)
    for layer in layer_iter:
        patcher = patchers[layer]
        cache = patcher.baselines(setup)

        # 1. Prune to features that actually fire.
        active = patcher.active_features(setup)
        if not active:
            continue

        # 2. Rank by attribution (approximate, one backward pass).
        scores = attribution_scores(patcher, setup)
        active_t = torch.as_tensor(active, dtype=torch.long)
        ranked = active_t[scores[active_t].abs().argsort(descending=True)]
        shortlist = ranked[:top_k_attribution].tolist()

        # 3. Confirm with exact greedy path patching.
        chosen, _ = greedy_select(
            patcher,
            setup,
            shortlist,
            max_features=max_features_per_layer,
            min_gain=min_gain,
            cache=cache,
            progress=progress,
        )
        if not chosen:
            continue
        selected_by_layer[layer] = chosen

        # 4. Localise each chosen feature to positions.
        for feature in chosen:
            solo = patcher.direct_causal_effect(setup, feature, cache=cache)
            effects = positionwise_effect(patcher, setup, feature, cache=cache, progress=progress)

            peak = float(effects.abs().max())
            if peak < 1e-8:
                token_positions: list[int] = []
                position_effects: list[float] = []
            else:
                keep = (effects.abs() >= position_threshold * peak).nonzero(as_tuple=True)[0]
                # Position 0 is <cls>, which is not a residue and cannot be
                # mapped onto a structure; drop it rather than emitting a
                # residue number that does not exist.
                keep = [int(p) for p in keep.tolist() if p > 0]
                token_positions = keep
                position_effects = [float(effects[p]) for p in keep]

            nodes.append(
                CircuitNode(
                    layer=layer,
                    feature=feature,
                    dce=solo.dce,
                    normalized_dce=solo.normalized_dce,
                    token_positions=token_positions,
                    position_effects=position_effects,
                )
            )

    # 5. Joint recovery across every selected feature, layer by layer together.
    joint_metric = _joint_patch_metric(patchers, setup, selected_by_layer)

    edges = (
        discover_edges(patchers, setup, nodes, threshold=edge_threshold, progress=progress)
        if find_edges and len(nodes) > 1
        else []
    )

    any_patcher = patchers[layer_iter[0]]
    base = any_patcher.baselines(setup)
    return Circuit(
        nodes=nodes,
        edges=edges,
        model_name=getattr(any_patcher.wrapper.model.config, "name_or_path", "unknown"),
        sequence="".join(setup.clean.sequences[:1]),
        mutation=setup.mutation.raw,
        clean_metric=base["clean_metric"],
        corrupted_metric=base["corrupted_metric"],
        joint_metric=joint_metric,
        config={
            "layers": layer_iter,
            "top_k_attribution": top_k_attribution,
            "max_features_per_layer": max_features_per_layer,
            "min_gain": min_gain,
            "position_threshold": position_threshold,
            "splice_mode": any_patcher.splice_mode,
            "selected_by_layer": {str(k): v for k, v in selected_by_layer.items()},
        },
    )


@torch.no_grad()
def _joint_patch_metric(
    patchers: dict[int, CausalPatcher],
    setup: PatchingSetup,
    selected_by_layer: dict[int, list[int]],
) -> float:
    """Patch every selected feature at every layer in a single forward pass.

    All the hooks are registered together so the layers interact exactly as they
    would in the full circuit, rather than being measured one at a time and
    summed — which would assume an additivity the model does not have.
    """
    if not selected_by_layer:
        base = patchers[next(iter(patchers))].baselines(setup)
        return base["corrupted_metric"]

    import contextlib as _contextlib

    with _contextlib.ExitStack() as stack:
        model = None
        for layer, features in selected_by_layer.items():
            patcher = patchers[layer]
            model = patcher.model
            cache = patcher.baselines(setup)
            stack.enter_context(
                patcher._patched(cache["clean_latents"], features, None)
            )
        assert model is not None
        logits = model(**setup.corrupted.as_model_kwargs()).logits
        return float(setup.metric(logits))
