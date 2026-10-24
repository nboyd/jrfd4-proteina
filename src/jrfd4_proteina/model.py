"""The RFD4-Proteina flow network (`LocalLatentsTransformer`) and AE decoder (`DecoderTransformer`)."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jrfd4_proteina import constants as C
from jrfd4_proteina.backend import from_torch, register_from_torch, stack_modules
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.features import (
    DecoderInputs,
    FeatureFactory,
    FlowInputs,
    ZeroFeatures,
)
from jrfd4_proteina.layers import (
    AdaptiveLayerNorm,
    AttnAndTransition,
    Linear,
    Norm,
    PairReprUpdate,
    Transition,
)


class Trunk(eqx.Module):
    """Attention/transition layers with a pair update after every `every`-th layer, scanned in blocks.

    Block b runs layers `b * every ... b * every + every - 1`, with the pair update right after its
    first layer (the reference updates pairs after layers `i % every == 0`, `i < nlayers - 1`).
    """

    attn: AttnAndTransition  # arrays stacked as [n_blocks, every, ...]
    attn_static: AttnAndTransition = eqx.field(static=True)
    pair: PairReprUpdate | None  # arrays stacked as [n_blocks, ...]
    pair_static: PairReprUpdate | None = eqx.field(static=True)
    every: int = eqx.field(static=True)

    @classmethod
    def from_torch(cls, attn_layers, pair_layers, every: int) -> Trunk:
        attn = [from_torch(layer) for layer in attn_layers]
        n_blocks = len(attn) // every
        stacked, attn_static = stack_modules(attn)
        stacked = jax.tree.map(lambda x: x.reshape((n_blocks, every) + x.shape[1:]), stacked)
        if pair_layers is None:
            return cls(stacked, attn_static, None, None, every)
        expected = [i for i in range(len(attn) - 1) if i % every == 0]
        present = [i for i, layer in enumerate(pair_layers) if layer is not None]
        if len(attn) % every or present != expected or expected != list(range(0, len(attn), every)):
            raise NotImplementedError("pair updates must fall on the first layer of equal-size blocks")
        pair, pair_static = stack_modules([from_torch(pair_layers[i]) for i in present])
        return cls(stacked, attn_static, pair, pair_static, every)

    def __call__(
        self, s: Float[Array, "N D"], z: Float[Array, "N N P"] | None, c: Float[Array, "N C"], mask: Bool[Array, " N"]
    ) -> tuple[Float[Array, "N D"], Float[Array, "N N P"] | None]:
        @jax.checkpoint
        def block(carry, params):
            s, z = carry
            attn_params, pair_params = params
            for i in range(self.every):
                layer = eqx.combine(jax.tree.map(lambda x, i=i: x[i], attn_params), self.attn_static)
                s = layer(s, z, c, mask)
                if i == 0 and pair_params is not None:
                    z = eqx.combine(pair_params, self.pair_static)(s, z, mask)
            return (s, z), None

        (s, z), _ = jax.lax.scan(block, (s, z), (self.attn, self.pair))
        return s, z


class NormLinear(eqx.Module):
    """`Sequential(norm, linear)` heads."""

    norm: Norm
    linear: Linear

    @classmethod
    def from_torch(cls, sequential) -> NormLinear:
        return cls(from_torch(sequential[0]), from_torch(sequential[1]))

    def __call__(self, x):
        return self.linear(self.norm(x))


@register_from_torch("rfproteina.nn.modules.pair_rep_initial.PairReprBuilder")
class PairReprBuilder(eqx.Module):
    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory | None
    adaln: AdaptiveLayerNorm | None

    @classmethod
    def from_torch(cls, m) -> PairReprBuilder:
        has_cond = m.cond_factory is not None
        return cls(
            from_torch(m.init_repr_factory),
            from_torch(m.cond_factory) if has_cond else None,
            from_torch(m.adaln) if has_cond else None,
        )

    def __call__(self, batch: Batch, inputs: FlowInputs) -> Float[Array, "N N P"]:
        z = self.init_repr_factory(batch, inputs)
        if self.cond_factory is None:
            return z
        mask = batch.padding_mask
        return self.adaln(z, self.cond_factory(batch, inputs), mask[:, None] & mask[None, :])


class FlowOutput(eqx.Module):
    v_bb_ca: Float[Array, "N 3"]  # velocity, nm per unit time
    v_local_latents: Float[Array, "N L"]
    sequence_logits: Float[Array, "N V"]
    insertion_logits: Float[Array, "N N"]


def flow_emit_masks(batch: Batch) -> tuple[Bool[Array, " N"], Bool[Array, " N"]]:
    """Tokens that carry each flow modality: (bb_ca, local_latents)."""
    bb_ca = batch.padding_mask & ~batch.is_in_context_conditioning
    return bb_ca, bb_ca & batch.supports_local_latent


@register_from_torch("rfproteina.nn.local_latents_transformer.LocalLatentsTransformer")
class LocalLatentsTransformer(eqx.Module):
    """The flow network: predicts CA and local-latent velocities (plus sequence / insertion logits)."""

    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory
    transition_c_1: Transition
    transition_c_2: Transition
    pair_repr_builder: PairReprBuilder
    trunk: Trunk
    s_cycle_norm: Norm
    s_cycle_proj: Linear
    z_cycle_norm: Norm
    z_cycle_proj: Linear
    local_latents_linear: NormLinear
    ca_linear: NormLinear
    sequence_readout: NormLinear
    insertion_q_proj: Linear
    insertion_k_proj: Linear
    recycle_dim: int | None = eqx.field(static=True)  # width of the remaining-passes feature, if conditioned

    @classmethod
    def from_torch(cls, m) -> LocalLatentsTransformer:
        if not (m.recycle_enabled and m.update_pair_repr and m.sequence_readout_enabled):
            raise NotImplementedError("only the released configuration (recycling, pair updates, readout) is ported")
        recycle_feats = [f for f in m.cond_factory.feat_creators if type(f).__name__ == "RecycleRemainingBatchFeat"]
        return cls(
            init_repr_factory=from_torch(m.init_repr_factory),
            cond_factory=from_torch(m.cond_factory),
            transition_c_1=from_torch(m.transition_c_1),
            transition_c_2=from_torch(m.transition_c_2),
            pair_repr_builder=from_torch(m.pair_repr_builder),
            trunk=Trunk.from_torch(m.transformer_layers, m.pair_update_layers, m.update_pair_repr_every_n),
            s_cycle_norm=from_torch(m.s_cycle_norm),
            s_cycle_proj=from_torch(m.s_cycle_proj),
            z_cycle_norm=from_torch(m.z_cycle_norm),
            z_cycle_proj=from_torch(m.z_cycle_proj),
            local_latents_linear=NormLinear.from_torch(m.local_latents_linear),
            ca_linear=NormLinear.from_torch(m.ca_linear),
            sequence_readout=NormLinear.from_torch(m.sequence_readout),
            insertion_q_proj=from_torch(m.insertion_q_proj),
            insertion_k_proj=from_torch(m.insertion_k_proj),
            recycle_dim=recycle_feats[0].dim if (recycle_feats and m.condition_on_recycle_remaining) else None,
        )

    def conditioning(self, batch: Batch, inputs: FlowInputs) -> Float[Array, "N C"]:
        c = self.cond_factory(batch, inputs)
        return self.transition_c_2(self.transition_c_1(c, batch.padding_mask), batch.padding_mask)

    def __call__(self, batch: Batch, inputs: FlowInputs, n_recycle: int) -> FlowOutput:
        """Run `n_recycle` trunk passes (at least one); gradients flow through the last pass only."""
        mask = batch.padding_mask
        c_first = self.conditioning(batch, inputs)
        s_init = self.init_repr_factory(batch, inputs) * mask[:, None]
        z_init = self.pair_repr_builder(batch, inputs)
        n_recycle = max(1, n_recycle)
        condition_remaining = self.recycle_dim is not None and n_recycle > 1

        def trunk_pass(cycle: Int[Array, ""], s, z):
            if condition_remaining:
                remaining = jax.nn.one_hot(jnp.minimum(n_recycle - cycle - 1, self.recycle_dim - 1), self.recycle_dim)
                c_later = self.conditioning(batch, dataclasses.replace(inputs, recycle_remaining=remaining))
                c = jnp.where(cycle == 0, c_first, c_later)
            else:
                c = c_first
            z_in = z_init + self.z_cycle_proj(self.z_cycle_norm(z))
            s_in = s_init + self.s_cycle_proj(self.s_cycle_norm(s))
            return self.trunk(s_in, z_in, c, mask)

        s, z = jnp.zeros_like(s_init), jnp.zeros_like(z_init)
        s, z = jax.lax.fori_loop(
            0, n_recycle - 1, lambda cycle, sz: jax.lax.stop_gradient(trunk_pass(cycle, *sz)), (s, z)
        )
        s, _ = trunk_pass(jnp.array(n_recycle - 1), s, z)

        emit_bb_ca, emit_latents = flow_emit_masks(batch)
        s_detached = jax.lax.stop_gradient(s)
        q, k = self.insertion_q_proj(s_detached), self.insertion_k_proj(s_detached)
        return FlowOutput(
            v_bb_ca=self.ca_linear(s) * emit_bb_ca[:, None],
            v_local_latents=self.local_latents_linear(s) * emit_latents[:, None],
            sequence_logits=self.sequence_readout(s_detached) * mask[:, None],
            insertion_logits=q @ k.T * s.shape[-1] ** -0.5,
        )


class DecoderOutput(eqx.Module):
    coords_nm: Float[Array, "N 37 3"]
    residue_type: Int[Array, " N"]
    atom_mask: Bool[Array, "N 37"]
    emit_mask: Bool[Array, " N"]
    seq_logits: Float[Array, "N V"]


def ae_masks(batch: Batch) -> tuple[Bool[Array, " N"], Bool[Array, " N"]]:
    """The decoder's (attention, emit) token masks."""
    attention = batch.padding_mask & ~batch.is_in_context_conditioning
    return attention, attention & batch.supports_local_latent


@register_from_torch("rfproteina.partial_autoencoder.decoder.DecoderTransformer")
class DecoderTransformer(eqx.Module):
    """AE decoder: (CA, local latent, token conditioning) -> residue type and atom37 coordinates."""

    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory | ZeroFeatures
    transition_c_1: Transition
    transition_c_2: Transition
    trunk: Trunk
    logit_linear: NormLinear
    struct_linear: NormLinear
    atom_presence_linear: NormLinear
    allowed_residue_types: Bool[Array, "T V"]
    residue_atom_mask: Bool[Array, "V 37"]

    @classmethod
    def from_torch(cls, m) -> DecoderTransformer:
        if m.use_attn_pair_bias or m.update_pair_repr or m.abs_coors:
            raise NotImplementedError("only the released decoder (no pair track, CA-relative coordinates) is ported")
        return cls(
            init_repr_factory=from_torch(m.init_repr_factory),
            cond_factory=from_torch(m.cond_factory),
            transition_c_1=from_torch(m.transition_c_1),
            transition_c_2=from_torch(m.transition_c_2),
            trunk=Trunk.from_torch(m.transformer_layers, None, 1),
            logit_linear=NormLinear.from_torch(m.logit_linear),
            struct_linear=NormLinear.from_torch(m.struct_linear),
            atom_presence_linear=NormLinear.from_torch(m.atom_presence_linear),
            allowed_residue_types=from_torch(m.allowed_residue_types),
            residue_atom_mask=from_torch(m.residue_atom_mask),
        )

    def __call__(
        self, batch: Batch, inputs: DecoderInputs, required_atom_mask: Bool[Array, "N 37"] | None = None
    ) -> DecoderOutput:
        """Decode, keeping conditioned residue types; `required_atom_mask` forces atom slots on."""
        attention, emit = ae_masks(batch)
        c = self.cond_factory(batch, inputs)
        c = self.transition_c_2(self.transition_c_1(c, attention), attention)
        s = self.init_repr_factory(batch, inputs) * attention[:, None]
        s, _ = self.trunk(s, None, c, attention)

        logits = self.logit_linear(s) * emit[:, None]
        allowed = self.allowed_residue_types[batch.token_type] | ~emit[:, None]
        logits = jnp.where(allowed, logits, jnp.finfo(logits.dtype).min)

        offsets = (self.struct_linear(s) * emit[:, None]).reshape(-1, 37, 3)
        coords = (offsets.at[:, 1].set(0.0) + inputs.ca_coors_nm[:, None]) * emit[:, None, None]

        predicted = jnp.argmax(logits, -1) * emit
        conditioned = (batch.cond_sequence != C.MASK_RESIDUE) & emit
        residue_type = jnp.where(conditioned, batch.cond_sequence, predicted).astype(jnp.int32)
        present = (self.atom_presence_linear(s) > 0).at[:, 1].set(True)
        if required_atom_mask is not None:
            present = present | required_atom_mask
        atom_mask = self.residue_atom_mask[residue_type] & present & emit[:, None]
        return DecoderOutput(
            coords_nm=coords * atom_mask[..., None],
            residue_type=residue_type * emit,
            atom_mask=atom_mask,
            emit_mask=emit,
            seq_logits=logits,
        )
