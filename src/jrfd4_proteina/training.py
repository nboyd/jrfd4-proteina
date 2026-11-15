"""Flow-matching fine-tuning of the flow network through LoRA adapters (the reference's `Proteina.training_step`
and `ProductSpaceFlowMatcher.compute_loss`).

One training example is corrupted at per-modality times `t` (`x_t = (1 - t) x_0 + t x_1`, with `x_1` the CA trace
and the frozen AE encoder's latents) and the network's clean prediction `x_t + (1 - t) v` is regressed on `x_1`.
Everything is single-example and pure; `batch_loss` maps over a stacked batch and `train_step` is the jitted
update. The model and optimizer are arguments, never closed over.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jrfd4_proteina import constants as C
from jrfd4_proteina import lora
from jrfd4_proteina.batch import Arr, Batch
from jrfd4_proteina.features import FlowInputs
from jrfd4_proteina.model import FlowOutput, LocalLatentsTransformer
from jrfd4_proteina.sampling import RFD4Proteina, State, corrupt, draw_noise, encode

# --------------------------------------------------------------------------- data


class Targets(eqx.Module):
    """Supervision for one example, beside the structure in `Batch.coords`."""

    residue_type: Int[Arr, " N"]  # unified vocabulary; -1 on padding
    is_resolved: Bool[Arr, " N"]  # every atom of the token is resolved
    loss_weight: Float[Arr, " N"]  # per-token weight of the flow-matching losses
    insertion_source_idx: Int[Arr, " N"]  # for unindexed tokens, the indexed token they copy; -1 elsewhere
    tetrahedral_vertices: Int[Arr, "N 4"]  # vertex tokens of the tetrahedron at a chiral centre; -1 where absent

    @classmethod
    def from_reference(cls, arrays: Mapping[str, np.ndarray]) -> Targets:
        return cls(
            residue_type=np.asarray(arrays["residue_type"]),
            is_resolved=np.asarray(arrays["is_resolved_mask"]),
            loss_weight=np.asarray(arrays["loss_weight"]),
            insertion_source_idx=np.asarray(arrays["insertion_source_idx"]),
            tetrahedral_vertices=np.asarray(arrays["tetrahedral_vertices"]),
        )


class Example(eqx.Module):
    batch: Batch
    targets: Targets


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class MixUniformBeta:
    """Times from `Beta(a, b)`, or with probability `p_uniform` from `U(0, 1)` (the reference's `mix_unif_beta`)."""

    a: float
    b: float
    p_uniform: float

    def sample(self, key: PRNGKeyArray) -> Float[Array, ""]:
        k_beta, k_uniform, k_mix = jax.random.split(key, 3)
        uniform = jax.random.uniform(k_mix) < self.p_uniform
        return jnp.where(uniform, jax.random.uniform(k_uniform), jax.random.beta(k_beta, self.a, self.b))


class Recycling(eqx.Module):
    """One step's trunk recycling, shared by its examples."""

    depth: Int[Array, ""]  # trunk passes
    condition_on_remaining: Bool[Array, ""]  # whether later passes are told how many remain


@dataclass(frozen=True)
class PoissonRecycling:
    """`Poisson(mean)` trunk passes clipped to `[1, maximum]`; with probability `condition_probability` the
    passes after the first are told how many remain (the checkpoint's `recycle.condition_probability`)."""

    mean: float
    maximum: int
    condition_probability: float

    def sample(self, key: PRNGKeyArray) -> Recycling:
        k_depth, k_condition = jax.random.split(key)
        depth = jnp.clip(jax.random.poisson(k_depth, self.mean), 1, self.maximum)
        return Recycling(depth, jax.random.uniform(k_condition) < self.condition_probability)


@dataclass(frozen=True)
class TrainingConfig:
    t_bb_ca: MixUniformBeta
    t_local_latents: MixUniformBeta
    recycle: PoissonRecycling
    # Noise candidates per example: the one with the lowest flow-matching loss is trained on (1: off)
    exploration_k: int
    signed_volume_weight: float
    sequence_weight: float = 1.0


def release_training_config() -> TrainingConfig:
    """The reference's fine-tuning recipe (`experiment/sota_finetune`)."""
    return TrainingConfig(
        t_bb_ca=MixUniformBeta(1.9, 1.0, 0.02),
        t_local_latents=MixUniformBeta(1.0, 1.5, 0.2),
        recycle=PoissonRecycling(mean=3.0, maximum=9, condition_probability=0.5),
        exploration_k=4,
        signed_volume_weight=2.5,
    )


# --------------------------------------------------------------------------- losses


class Losses(eqx.Module):
    """Per-example loss terms (already weighted) and diagnostics."""

    bb_ca: Float[Array, ""]
    local_latents: Float[Array, ""]
    sequence: Float[Array, ""]
    insertion: Float[Array, ""]
    signed_volume: Float[Array, ""]
    sequence_accuracy: Float[Array, ""]  # not optimised
    insertion_accuracy: Float[Array, ""]

    @property
    def total(self) -> Float[Array, ""]:
        return self.bb_ca + self.local_latents + self.sequence + self.insertion + self.signed_volume


def flow_matching_loss(
    predicted: Float[Array, "N D"],
    clean: Float[Array, "N D"],
    t: Float[Array, ""],
    supervised: Bool[Array, " N"],
    weight: Float[Array, " N"],
) -> Float[Array, ""]:
    """Weighted squared error of the clean prediction per supervised token, scaled by `1 / (1 - t)^2` (the
    velocity error)."""
    error = jnp.where(supervised[:, None], clean - predicted, 0.0)
    per_token = (error**2).sum(-1) * weight
    return per_token.sum() / jnp.maximum(supervised.sum(), 1) / ((1.0 - t) ** 2 + 1e-5)


def _cross_entropy(logits: Float[Array, "N K"], target: Int[Array, " N"], mask: Bool[Array, " N"]):
    """Mean cross entropy and accuracy over `mask` (zero when it is empty)."""
    target = jnp.where(mask, target, 0)
    nll = -jnp.take_along_axis(jax.nn.log_softmax(logits.astype(jnp.float32)), target[:, None], -1)[:, 0]
    count = jnp.maximum(mask.sum(), 1)
    correct = (jnp.argmax(logits, -1) == target) & mask
    return jnp.where(mask, nll, 0.0).sum() / count, correct.sum() / count


def sequence_loss(logits: Float[Array, "N V"], batch: Batch, targets: Targets):
    """Cross entropy of the sequence readout on supervised tokens whose residue type is not given."""
    mask = batch.latent_mask & targets.is_resolved & (batch.cond_sequence == C.MASK_RESIDUE)
    return _cross_entropy(logits, targets.residue_type, mask)


def insertion_loss(logits: Float[Array, "N N"], batch: Batch, targets: Targets):
    """Cross entropy of each unindexed token's choice, among the indexed tokens, of the one it copies."""
    source = targets.insertion_source_idx
    indexed = batch.padding_mask & (source < 0)
    return _cross_entropy(jnp.where(indexed[None, :], logits, -1e9), source, source >= 0)


def signed_volume_loss(
    predicted: Float[Array, "N 3"],
    reference: Float[Array, "N 3"],
    tetrahedra: Int[Array, "N 4"],
    supervised: Bool[Array, " N"],
) -> Float[Array, ""]:
    """Squared hinge on the handedness of the tetrahedra at chiral centres.

    Signed volumes are normalised by the reference edge lengths. A tetrahedron counts if its centre and vertices
    are supervised and its normalised reference volume is at least 0.1; the predicted volume should then have the
    reference's sign by a margin of `min(0.2, half that volume)`.
    """
    vertices = jnp.maximum(tetrahedra, 0)
    p, r = predicted[vertices], jax.lax.stop_gradient(reference)[vertices]
    eligible = (tetrahedra[:, 0] >= 0) & supervised & supervised[vertices].all(-1) & jnp.isfinite(r).all((-1, -2))
    p, r = (jnp.where(eligible[:, None, None], x, 0.0) for x in (p, r))
    pv, rv = p[:, :3] - p[:, 3:], r[:, :3] - r[:, 3:]
    volume = lambda v: (v[:, 0] * jnp.cross(v[:, 1], v[:, 2])).sum(-1)
    normalizer = jnp.maximum(jnp.linalg.norm(rv, axis=-1), 0.01).prod(-1)
    normalized_reference = volume(rv) / normalizer
    eligible = eligible & (jnp.abs(normalized_reference) >= 0.1)
    margin = jnp.minimum(jnp.abs(normalized_reference) * 0.5, 0.2)
    penalty = jax.nn.relu(margin - jnp.sign(normalized_reference) * volume(pv) / normalizer) ** 2
    return jnp.where(eligible, penalty, 0.0).sum() / jnp.maximum(eligible.sum(), 1)


class Corruption(eqx.Module):
    """One noised training example: the flow network's inputs and its regression target."""

    clean: State  # x_1, zero outside each modality's emit mask
    noisy: State  # x_t
    t: Float[Array, " 2"]  # (bb_ca, local_latents)
    coord_group_priority: Float[Array, " K"]


def corrupt_example(model: RFD4Proteina, batch: Batch, t: Float[Array, " 2"], key: PRNGKeyArray) -> Corruption:
    """Encode the example's structure (a posterior draw of the latents) and noise it to times `t`."""
    k_encode, k_noise = jax.random.split(key)
    clean = encode(model.encoder, batch, key=k_encode)
    clean = State(clean.bb_ca * batch.design_mask[:, None], clean.local_latents)
    noise = draw_noise(k_noise, batch, nsteps=1, latent_dim=model.latent_dim)
    noisy = corrupt(batch, clean, t, noise, model.noise_config)
    return Corruption(clean, noisy, t, noise.steps.coord_group_priority[0])


def losses(
    out: FlowOutput, example: Example, corruption: Corruption, config: TrainingConfig
) -> Losses:
    """Every loss term of one example, from the network output at its corruption."""
    batch, targets = example.batch, example.targets
    t_bb, t_lat = corruption.t
    clean, noisy = corruption.clean, corruption.noisy
    emit_bb, emit_lat = batch.design_mask, batch.latent_mask
    predicted_bb = (noisy.bb_ca + (1.0 - t_bb) * out.v_bb_ca) * emit_bb[:, None]
    predicted_lat = (noisy.local_latents + (1.0 - t_lat) * out.v_local_latents) * emit_lat[:, None]
    supervised_bb, supervised_lat = emit_bb & targets.is_resolved, emit_lat & targets.is_resolved

    sequence, sequence_accuracy = sequence_loss(out.sequence_logits, batch, targets)
    insertion, insertion_accuracy = insertion_loss(out.insertion_logits, batch, targets)
    chiral = signed_volume_loss(
        predicted_bb, clean.bb_ca, targets.tetrahedral_vertices, supervised_bb & (targets.loss_weight > 0)
    )
    return Losses(
        bb_ca=flow_matching_loss(predicted_bb, clean.bb_ca, t_bb, supervised_bb, targets.loss_weight),
        local_latents=flow_matching_loss(
            predicted_lat, clean.local_latents, t_lat, supervised_lat, targets.loss_weight
        ),
        sequence=config.sequence_weight * sequence,
        insertion=insertion,
        signed_volume=config.signed_volume_weight * chiral,
        sequence_accuracy=sequence_accuracy,
        insertion_accuracy=insertion_accuracy,
    )


def denoising_losses(
    flow: LocalLatentsTransformer,
    example: Example,
    corruption: Corruption,
    recycling: Recycling,
    config: TrainingConfig,
) -> Losses:
    """Evaluate the flow network at `corruption` and score its output."""
    c = corruption
    inputs = FlowInputs(c.noisy.bb_ca, c.noisy.local_latents, c.t[0], c.t[1], c.coord_group_priority)
    out = flow(example.batch, inputs, recycling.depth, recycling.condition_on_remaining)
    return losses(out, example, c, config)


# --------------------------------------------------------------------------- objective


def example_loss(
    adapters: lora.Adapters,
    model: RFD4Proteina,
    example: Example,
    key: PRNGKeyArray,
    recycling: Recycling,
    config: TrainingConfig,
) -> tuple[Float[Array, ""], Losses]:
    """The training loss of one example with `adapters` attached to `model.flow`.

    With `config.exploration_k > 1`, that many corruptions sharing `t` are scored without gradients and the loss
    is that of the one with the lowest flow-matching error.
    """
    flow = lora.attach(model.flow, adapters)
    k_bb, k_lat, k_corrupt = jax.random.split(key, 3)
    t = jnp.stack([config.t_bb_ca.sample(k_bb), config.t_local_latents.sample(k_lat)])
    keys = jax.random.split(k_corrupt, config.exploration_k)
    candidates = jax.vmap(lambda k: corrupt_example(model, example.batch, t, k))(keys)
    winner = 0
    if config.exploration_k > 1:
        frozen = jax.lax.stop_gradient(flow)

        def score(corruption: Corruption) -> Float[Array, ""]:
            candidate = denoising_losses(frozen, example, corruption, recycling, config)
            return candidate.bb_ca + candidate.local_latents

        winner = jnp.argmin(jax.vmap(score)(candidates))
    corruption = jax.lax.stop_gradient(jax.tree.map(lambda x: x[winner], candidates))
    terms = denoising_losses(flow, example, corruption, recycling, config)
    return terms.total, terms


def batch_loss(
    adapters: lora.Adapters, model: RFD4Proteina, examples: Example, key: PRNGKeyArray, config: TrainingConfig
) -> tuple[Float[Array, ""], Losses]:
    """Mean loss over stacked `examples` (leading axis B) under one sampled `Recycling`; also the mean terms."""
    k_recycle, k_examples = jax.random.split(key)
    recycling = config.recycle.sample(k_recycle)
    keys = jax.random.split(k_examples, examples.batch.padding_mask.shape[0])
    per_example = jax.vmap(lambda example, k: example_loss(adapters, model, example, k, recycling, config))
    loss, terms = per_example(examples, keys)
    return loss.mean(), jax.tree.map(jnp.mean, terms)


@eqx.filter_jit
def train_step(
    adapters: lora.Adapters,
    opt_state,
    model: RFD4Proteina,
    optimizer,
    examples: Example,
    key: PRNGKeyArray,
    config: TrainingConfig,
):
    """One optimizer update of `adapters`. `optimizer` is an optax `GradientTransformation`, e.g.
    `jrfd4_proteina.optim.optimizer()`. Returns `(adapters, opt_state, mean Losses)`."""
    (_, terms), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(adapters, model, examples, key, config)
    updates, opt_state = optimizer.update(grads, opt_state, adapters)
    return eqx.apply_updates(adapters, updates), opt_state, terms


def ema(average: lora.Adapters, adapters: lora.Adapters, decay: float = 0.999) -> lora.Adapters:
    """Exponential moving average of the adapters (start it from the adapters themselves)."""
    return jax.tree.map(lambda avg, new: decay * avg + (1.0 - decay) * new, average, adapters)
