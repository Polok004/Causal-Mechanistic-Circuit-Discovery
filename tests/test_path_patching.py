"""Tests for hooks, the patching algebra and circuit extraction.

These are the tests that catch the bugs that matter. A patching pipeline fails
*quietly*: every number it produces is finite and plausible, so a sign error or
a hook that never gets removed shows up as a wrong conclusion rather than a
crash. The properties asserted here are the ones with known failure modes:

* patching a feature to its own value must be a no-op (identity);
* the corrupted baseline must be measured with no hook attached, or every
  direct effect is exactly zero;
* hooks must be removed even when the forward pass raises;
* error-preserving splicing must cancel SAE reconstruction error, while the
  naive splice must not;
* attribution patching must agree in sign with exact patching, since its only
  job is to rank.
"""

from __future__ import annotations

import pytest
import torch

from interpretability.path_patching import (
    CausalPatcher,
    LogitDiffMetric,
    PatchingSetup,
    attribution_scores,
    positionwise_effect,
)
from models.esm_hooks import (
    ESMActivationExtractor,
    HookHandleSet,
    resolve_encoder_layers,
)
from utils.protein import parse_mutation, seq_to_token_pos, token_to_seq_pos

# --------------------------------------------------------------------------- #
# Hooks                                                                         #
# --------------------------------------------------------------------------- #


def test_resolve_encoder_layers_finds_blocks(tiny_model):
    layers = resolve_encoder_layers(tiny_model)
    assert len(layers) == tiny_model.config.num_hidden_layers


def test_resolve_encoder_layers_reports_what_it_found():
    """A layout change must produce a diagnostic, not an AttributeError."""
    import torch.nn as nn

    class NotAnEsm(nn.Module):
        def __init__(self):
            super().__init__()
            self.something_else = nn.Linear(2, 2)

    with pytest.raises(AttributeError, match="top-level children"):
        resolve_encoder_layers(NotAnEsm())


def test_extractor_captures_expected_shapes(wrapper, sequences):
    batch = wrapper.tokenize(sequences)
    acts = wrapper.residual_stream(batch, [0, 2])
    assert set(acts) == {0, 2}
    for tensor in acts.values():
        assert tensor.shape == (len(sequences), batch.input_ids.shape[1], wrapper.d_model)


def test_extractor_removes_hooks_on_exit(wrapper, sequences):
    layer = resolve_encoder_layers(wrapper.model)[1]
    before = len(layer._forward_hooks)
    with ESMActivationExtractor(wrapper.model, [1]):
        assert len(layer._forward_hooks) == before + 1
    assert len(layer._forward_hooks) == before


def test_hooks_removed_even_when_forward_raises(wrapper, sequences):
    """A hook leaked by an exception corrupts every later forward pass."""
    from models.esm_hooks import make_patch_hook

    layer = resolve_encoder_layers(wrapper.model)[1]
    before = len(layer._forward_hooks)

    def boom(_hidden):
        raise RuntimeError("intentional")

    handles = HookHandleSet()
    handles.add(layer.register_forward_hook(make_patch_hook(boom)))
    batch = wrapper.tokenize(sequences[:1])
    try:
        with handles:
            with pytest.raises(RuntimeError, match="intentional"):
                wrapper.model(**batch.as_model_kwargs())
    finally:
        pass
    assert len(layer._forward_hooks) == before


def test_flat_drops_padding(wrapper, sequences):
    batch = wrapper.tokenize(sequences)
    extractor = ESMActivationExtractor(wrapper.model, [1])
    with torch.no_grad(), extractor:
        wrapper.model(**batch.as_model_kwargs())
        flat = extractor.flat(1, batch.attention_mask)
    assert flat.shape[0] == int(batch.attention_mask.sum())
    assert flat.shape[0] < batch.input_ids.numel()  # padding really was present


def test_patch_hook_rejects_shape_change(wrapper, sequences):
    from models.esm_hooks import make_patch_hook

    layer = resolve_encoder_layers(wrapper.model)[1]
    handle = layer.register_forward_hook(make_patch_hook(lambda h: h[:, :-1]))
    batch = wrapper.tokenize(sequences[:1])
    try:
        with pytest.raises(ValueError, match="shape-preserving"):
            wrapper.model(**batch.as_model_kwargs())
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# Index conventions                                                             #
# --------------------------------------------------------------------------- #


def test_mutation_parsing_is_one_indexed():
    mutation = parse_mutation("A45T")
    assert mutation.wt_aa == "A"
    assert mutation.mut_aa == "T"
    assert mutation.seq_pos == 44          # 0-indexed
    assert mutation.one_indexed_pos == 45


def test_token_offset_roundtrip():
    for seq_pos in range(50):
        assert token_to_seq_pos(seq_to_token_pos(seq_pos)) == seq_pos


def test_cls_token_is_not_a_residue():
    with pytest.raises(ValueError, match="special token"):
        token_to_seq_pos(0)


def test_mutation_apply_checks_wildtype():
    mutation = parse_mutation("A2T")
    assert mutation.apply("MATGG") == "MTTGG"
    with pytest.raises(ValueError, match="expects wild-type"):
        parse_mutation("C2T").apply("MATGG")


def test_multi_substitution_is_rejected():
    with pytest.raises(ValueError, match="multi-substitution"):
        parse_mutation("A45T:G50S")


def test_setup_places_the_metric_at_the_mutated_token(wrapper, wt_sequence):
    setup = PatchingSetup.from_mutation(wrapper, wt_sequence, "M1A")
    assert setup.token_pos == 1  # residue 1 sits at token 1, after <cls>
    assert setup.clean.input_ids.shape == setup.corrupted.input_ids.shape
    # The two runs differ at exactly one token.
    diff = (setup.clean.input_ids != setup.corrupted.input_ids).sum()
    assert int(diff) == 1


# --------------------------------------------------------------------------- #
# Patching algebra                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def patcher(wrapper, sae):
    return CausalPatcher(wrapper, sae, layer_idx=1)


@pytest.fixture
def setup(wrapper, wt_sequence):
    mutation = parse_mutation(f"{wt_sequence[20]}21A") if wt_sequence[20] != "A" else parse_mutation(f"{wt_sequence[20]}21G")
    return PatchingSetup.from_mutation(wrapper, wt_sequence, mutation)


def test_patcher_rejects_mismatched_sae(wrapper):
    from models.sparse_autoencoder import SAEConfig, TopKSparseAutoencoder

    wrong_width = TopKSparseAutoencoder(SAEConfig(d_in=wrapper.d_model + 8, dict_mult=2, k=2))
    with pytest.raises(ValueError, match="different checkpoint"):
        CausalPatcher(wrapper, wrong_width, layer_idx=1)


def test_patcher_rejects_layer_mismatch(wrapper, sae):
    # `sae` is tagged layer_idx=1.
    with pytest.raises(ValueError, match="trained on layer 1"):
        CausalPatcher(wrapper, sae, layer_idx=2)


def test_patching_clean_into_clean_is_identity(patcher, wrapper, wt_sequence):
    """Patching a run with its own values must change nothing.

    This is the single most informative test in the file. It fails if the splice
    algebra is wrong, if activation scaling is applied asymmetrically, or if the
    hook writes into the wrong element of the layer's output tuple — three bugs
    that otherwise produce believable numbers.
    """
    setup = PatchingSetup.from_mutation(wrapper, wt_sequence, "M1A")
    with torch.no_grad():
        _, clean_latents = patcher.sae_latents(setup.clean)
        reference = patcher.run_metric(setup.clean, setup.metric)
        with patcher._patched(clean_latents, feature_idx=list(range(patcher.sae.d_sae)), positions=None):
            patched = float(
                setup.metric(patcher.model(**setup.clean.as_model_kwargs()).logits)
            )
    assert patched == pytest.approx(reference, abs=1e-4)


def test_error_preserving_splice_cancels_reconstruction_error(patcher, setup):
    """Patching *no* features must leave the forward pass untouched.

    With error-preserving splicing the two decodes are identical and cancel, so
    an empty patch is exactly a no-op. The naive "replace" mode instead
    substitutes the SAE reconstruction and changes the metric — which is the
    whole reason error-preserving splicing exists.
    """
    cache = patcher.baselines(setup)
    empty = patcher.direct_causal_effect(setup, [], cache=cache)
    assert empty.dce == pytest.approx(0.0, abs=1e-5)


def test_naive_splice_does_not_cancel_reconstruction_error(wrapper, sae, setup):
    """The contrast that justifies the default.

    An untrained SAE reconstructs poorly, so replacing the activation with its
    reconstruction must visibly move the metric even with nothing patched.
    """
    naive = CausalPatcher(wrapper, sae, layer_idx=1, splice_mode="replace")
    cache = naive.baselines(setup)
    empty = naive.direct_causal_effect(setup, [], cache=cache)
    assert abs(empty.dce) > 1e-4


def test_baseline_is_measured_without_hooks(patcher, setup):
    """Guards against the classic bug of computing the baseline under the hook.

    If the corrupted baseline were computed while the patch hook was live, it
    would equal the patched value and every DCE would be identically zero. Here
    we patch every feature at once, which must produce a large effect.
    """
    cache = patcher.baselines(setup)
    result = patcher.direct_causal_effect(
        setup, list(range(patcher.sae.d_sae)), cache=cache
    )
    assert result.corrupted_metric == pytest.approx(
        patcher.run_metric(setup.corrupted, setup.metric), abs=1e-5
    )
    assert abs(result.dce) > 1e-6


def test_patching_all_features_moves_the_metric_toward_clean(wrapper, trained_sae, setup):
    """Restoring the entire clean code should move the metric toward clean.

    Not all the way: the SAE is small and briefly trained, so its latents do not
    span the residual stream, and the clean and corrupted activations also
    differ in the component the dictionary does not capture. The assertion is
    therefore directional rather than quantitative.

    Uses the fitted dictionary: with random decoder directions the splice adds a
    vector unrelated to ``a_clean - a_corrupt`` and the sign carries no
    information.
    """
    patcher = CausalPatcher(wrapper, trained_sae, layer_idx=1)
    cache = patcher.baselines(setup)
    gap = cache["clean_metric"] - cache["corrupted_metric"]
    if abs(gap) < 1e-6:
        pytest.skip("this random-weight model does not respond to the mutation")
    result = patcher.direct_causal_effect(
        setup, list(range(patcher.sae.d_sae)), cache=cache
    )
    assert result.dce * gap > 0, "patching moved the metric away from the clean value"


def test_inactive_features_have_zero_effect(patcher, setup):
    """A feature that fires in neither run cannot do anything when patched."""
    active = set(patcher.active_features(setup))
    inactive = [f for f in range(patcher.sae.d_sae) if f not in active]
    if not inactive:
        pytest.skip("every feature was active")
    cache = patcher.baselines(setup)
    result = patcher.direct_causal_effect(setup, inactive[0], cache=cache)
    assert result.dce == pytest.approx(0.0, abs=1e-6)


def test_active_feature_pruning_is_sound(patcher, setup):
    """Pruning must never discard a feature that would have had an effect."""
    active = patcher.active_features(setup)
    assert 0 < len(active) <= patcher.sae.d_sae
    cache = patcher.baselines(setup)
    clean, corrupt = cache["clean_latents"], cache["corrupted_latents"]
    for f in range(min(patcher.sae.d_sae, 64)):
        fires = bool((clean[..., f].abs().max() > 0) or (corrupt[..., f].abs().max() > 0))
        assert fires == (f in active)


def test_normalized_dce_is_zero_when_there_is_no_gap():
    from interpretability.path_patching import PatchResult

    result = PatchResult(
        layer_idx=0,
        feature_idx=0,
        positions=(1,),
        clean_metric=1.0,
        corrupted_metric=1.0,
        patched_metric=5.0,
    )
    assert result.normalized_dce == 0.0


def test_sweep_is_sorted_by_absolute_effect(patcher, setup):
    features = patcher.active_features(setup)[:8]
    if len(features) < 2:
        pytest.skip("not enough active features")
    results = patcher.sweep_features(setup, features, progress=False)
    effects = [abs(r.dce) for r in results]
    assert effects == sorted(effects, reverse=True)


def test_positionwise_effects_have_one_entry_per_token(patcher, setup):
    features = patcher.active_features(setup)
    if not features:
        pytest.skip("no active features")
    effects = positionwise_effect(patcher, setup, features[0])
    assert effects.shape == (setup.seq_len,)


def test_ablation_changes_the_metric(patcher, setup):
    features = patcher.active_features(setup)[:4]
    if not features:
        pytest.skip("no active features")
    baseline = patcher.run_metric(setup.corrupted, setup.metric)
    ablated = patcher.ablate_features(setup.corrupted, setup.metric, features)
    assert ablated != pytest.approx(baseline, abs=1e-9)


# --------------------------------------------------------------------------- #
# Attribution patching                                                          #
# --------------------------------------------------------------------------- #


def test_attribution_scores_cover_the_dictionary(patcher, setup):
    scores = attribution_scores(patcher, setup)
    assert scores.shape == (patcher.sae.d_sae,)
    assert torch.isfinite(scores).all()


def test_attribution_is_zero_for_inactive_features(patcher, setup):
    """A feature with identical clean and corrupted values has zero delta.

    The attribution is ``delta * grad``, so this must hold exactly regardless of
    the gradient — a useful check that the delta is computed against the right
    tensor.
    """
    scores = attribution_scores(patcher, setup)
    active = set(patcher.active_features(setup))
    inactive = [f for f in range(patcher.sae.d_sae) if f not in active][:20]
    if not inactive:
        pytest.skip("every feature was active")
    assert torch.allclose(scores[inactive], torch.zeros(len(inactive)), atol=1e-6)


def test_attribution_agrees_in_sign_with_exact_patching(patcher, setup):
    """Attribution is used only for ranking, so sign agreement is what matters.

    Magnitudes are allowed to be wrong — the first-order approximation is known
    to be poor where the model saturates. The test is deliberately tolerant: it
    checks the strongest candidates only, and allows a minority to disagree.
    """
    scores = attribution_scores(patcher, setup)
    active = patcher.active_features(setup)
    if len(active) < 4:
        pytest.skip("not enough active features")

    active_t = torch.as_tensor(active)
    top = active_t[scores[active_t].abs().argsort(descending=True)][:6].tolist()

    cache = patcher.baselines(setup)
    agree = 0
    counted = 0
    for f in top:
        exact = patcher.direct_causal_effect(setup, f, cache=cache).dce
        approx = float(scores[f])
        if abs(exact) < 1e-7 or abs(approx) < 1e-7:
            continue
        counted += 1
        agree += (exact > 0) == (approx > 0)
    if counted == 0:
        pytest.skip("effects too small to compare signs")
    assert agree / counted >= 0.5


def test_attribution_leaves_no_hooks_behind(patcher, setup):
    layer = resolve_encoder_layers(patcher.model)[patcher.layer_idx]
    before = len(layer._forward_hooks)
    attribution_scores(patcher, setup)
    assert len(layer._forward_hooks) == before


def test_model_returns_to_eval_after_attribution(patcher, setup):
    patcher.model.eval()
    attribution_scores(patcher, setup)
    assert not patcher.model.training


# --------------------------------------------------------------------------- #
# Circuit extraction                                                            #
# --------------------------------------------------------------------------- #


def test_greedy_selection_is_monotonic(patcher, setup):
    """Recovery must not decrease as features are added."""
    from interpretability.circuit_extraction import greedy_select

    candidates = patcher.active_features(setup)[:12]
    if len(candidates) < 2:
        pytest.skip("not enough candidates")
    chosen, trajectory = greedy_select(
        patcher, setup, candidates, max_features=4, min_gain=0.0, progress=False
    )
    assert len(chosen) == len(trajectory)
    assert trajectory == sorted(trajectory)


def test_discover_circuit_produces_serialisable_output(patcher, setup, tmp_path):
    from interpretability.circuit_extraction import Circuit, discover_circuit

    circuit = discover_circuit(
        {1: patcher},
        setup,
        top_k_attribution=8,
        max_features_per_layer=2,
        find_edges=False,
        progress=False,
    )
    path = circuit.save(tmp_path / "circuit.json")
    reloaded = Circuit.load(path)

    assert reloaded.mutation == circuit.mutation
    assert len(reloaded.nodes) == len(circuit.nodes)
    assert reloaded.recovered_fraction == pytest.approx(circuit.recovered_fraction)


def test_circuit_residues_exclude_the_cls_token(patcher, setup):
    """Token 0 is <cls> and has no residue number; it must never be reported."""
    from interpretability.circuit_extraction import discover_circuit

    circuit = discover_circuit(
        {1: patcher},
        setup,
        top_k_attribution=8,
        max_features_per_layer=2,
        find_edges=False,
        progress=False,
    )
    for node in circuit.nodes:
        assert all(p > 0 for p in node.token_positions)
        assert all(r >= 1 for r in node.residues)


def test_circuit_recovered_fraction_is_defined_without_a_gap():
    from interpretability.circuit_extraction import Circuit

    circuit = Circuit(clean_metric=2.0, corrupted_metric=2.0, joint_metric=9.0)
    assert circuit.recovered_fraction == 0.0


# --------------------------------------------------------------------------- #
# Conservation                                                                  #
# --------------------------------------------------------------------------- #


def test_causal_conservation_full_patch_reaches_clean_behaviour(wrapper, wt_sequence):
    """Patching the *whole residual stream* must reproduce the clean run exactly.

    This is the conservation property the plan asks for, stated in the form that
    is actually exact. Summing individual feature effects does **not** equal the
    total effect — the model is nonlinear and features interact, and any test
    asserting additivity would be asserting something false. What must hold is
    that substituting the entire clean activation at a layer makes everything
    downstream of that layer identical to the clean forward pass.
    """
    from models.esm_hooks import make_patch_hook

    setup = PatchingSetup.from_mutation(wrapper, wt_sequence, "M1A")
    layer_idx = 1
    layers = resolve_encoder_layers(wrapper.model)

    with torch.no_grad():
        clean_acts = wrapper.residual_stream(setup.clean, [layer_idx])[layer_idx]
        clean_metric = float(
            setup.metric(wrapper.model(**setup.clean.as_model_kwargs()).logits)
        )

        handle = layers[layer_idx].register_forward_hook(
            make_patch_hook(lambda _hidden: clean_acts)
        )
        try:
            patched_metric = float(
                setup.metric(wrapper.model(**setup.corrupted.as_model_kwargs()).logits)
            )
        finally:
            handle.remove()

    assert patched_metric == pytest.approx(clean_metric, abs=1e-4)


def test_logit_diff_metric_is_shift_invariant(wrapper, wt_sequence):
    """A uniform shift of the logits must not change a logit *difference*."""
    setup = PatchingSetup.from_mutation(wrapper, wt_sequence, "M1A")
    with torch.no_grad():
        logits = wrapper.model(**setup.clean.as_model_kwargs()).logits
    metric = LogitDiffMetric(
        token_pos=setup.token_pos,
        wt_token_id=wrapper.aa_token_id("M"),
        mut_token_id=wrapper.aa_token_id("A"),
    )
    assert float(metric(logits)) == pytest.approx(float(metric(logits + 7.5)), abs=1e-4)
