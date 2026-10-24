"""Leaf and trunk layers of the RFD4-Proteina networks, for a single example.

Shapes: N tokens, D token channels, P pair channels, C conditioning channels, H heads.
"""

from __future__ import annotations

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from jrfd4_proteina.backend import AbstractFromTorch, from_torch, register_from_torch

NEG_INF = -1e4


@register_from_torch("torch.nn.modules.linear.Linear", "rfproteina.nn.modules.openfold_modules.primitives.Linear")
class Linear(AbstractFromTorch):
    weight: Float[Array, "Out In"]
    bias: Float[Array, " Out"] | None = None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        y = jnp.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias


@register_from_torch(
    "torch.nn.modules.normalization.LayerNorm", "rfproteina.nn.modules.openfold_modules.primitives.LayerNorm"
)
class LayerNorm(AbstractFromTorch):
    weight: Float[Array, " D"] | None = None
    bias: Float[Array, " D"] | None = None
    eps: float = 1e-5

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        mean = x.mean(-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(jnp.square(x - mean).mean(-1, keepdims=True) + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x if self.bias is None else x + self.bias


@register_from_torch("torch.nn.modules.normalization.RMSNorm", "rfproteina.nn.modules.norm.CompiledRMSNorm")
class RMSNorm(AbstractFromTorch):
    weight: Float[Array, " D"] | None = None
    eps: float = 1e-5

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)
        return x if self.weight is None else x * self.weight


Norm = LayerNorm | RMSNorm


@register_from_torch("torch.nn.modules.sparse.Embedding")
class Embedding(AbstractFromTorch):
    weight: Float[Array, "V D"]

    def __call__(self, index):
        return self.weight[index]


def masked(x: Float[Array, "N ..."], mask: Bool[Array, " N"]) -> Float[Array, "N ..."]:
    return x * mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))


@register_from_torch("rfproteina.nn.modules.seq_transition_af3.Transition")
class Transition(eqx.Module):
    """SwiGLU transition: `linear_out(silu(gate) * x)` with `[x, gate] = linear_in(norm(x))`."""

    linear_in: Linear
    linear_out: Linear
    norm: LayerNorm | None = None

    @classmethod
    def from_torch(cls, m):
        return cls(
            linear_in=from_torch(m.swish_linear[0]),
            linear_out=from_torch(m.linear_out),
            norm=from_torch(m.ln) if m.use_layer_norm else None,
        )

    def __call__(self, x: Float[Array, "N D"], mask: Bool[Array, " N"]) -> Float[Array, "N D"]:
        if self.norm is not None:
            x = self.norm(x)
        h, gate = jnp.split(self.linear_in(x), 2, axis=-1)
        return masked(self.linear_out(jax.nn.silu(gate) * h), mask)


@register_from_torch("rfproteina.nn.modules.adaptive_ln_scale.AdaptiveLayerNorm")
class AdaptiveLayerNorm(eqx.Module):
    norm: Norm
    norm_cond: Norm
    to_gamma: Linear
    to_beta: Linear

    @classmethod
    def from_torch(cls, m):
        return cls(
            norm=from_torch(m.norm),
            norm_cond=from_torch(m.norm_cond),
            to_gamma=from_torch(m.to_gamma[0]),
            to_beta=from_torch(m.to_beta),
        )

    def __call__(self, x: Float[Array, "*S D"], cond: Float[Array, "*S C"] | Float[Array, " C"], mask) -> Float[
        Array, "*S D"
    ]:
        cond = self.norm_cond(cond)
        out = self.norm(x) * jax.nn.sigmoid(self.to_gamma(cond)) + self.to_beta(cond)
        return out * mask[..., None]


@register_from_torch("rfproteina.nn.modules.adaptive_ln_scale.AdaptiveOutputScale")
class AdaptiveOutputScale(eqx.Module):
    to_gamma: Linear

    @classmethod
    def from_torch(cls, m):
        return cls(to_gamma=from_torch(m.to_adaln_zero_gamma[0]))

    def __call__(self, x: Float[Array, "N D"], cond: Float[Array, "N C"], mask: Bool[Array, " N"]):
        return masked(x * jax.nn.sigmoid(self.to_gamma(cond)), mask)


@register_from_torch("rfproteina.nn.modules.seq_transition_af3.TransitionADALN")
class TransitionADALN(AbstractFromTorch):
    adaln: AdaptiveLayerNorm
    transition: Transition
    scale_output: AdaptiveOutputScale

    def __call__(self, x: Float[Array, "N D"], cond: Float[Array, "N C"], mask: Bool[Array, " N"]):
        x = self.adaln(x, cond, mask)
        x = self.transition(x, mask)
        return masked(self.scale_output(x, cond, mask), mask)


@register_from_torch("rfproteina.nn.modules.pair_bias_attn.PairBiasAttention")
class PairBiasAttention(eqx.Module):
    """Gated multi-head attention with an optional pair bias; padded keys are masked."""

    node_norm: Norm
    to_qkv: Linear
    to_g: Linear
    to_out_node: Linear
    q_layer_norm: Norm | None
    k_layer_norm: Norm | None
    pair_norm: Norm | None
    to_bias: Linear | None
    heads: int = eqx.field(static=True)

    @classmethod
    def from_torch(cls, m):
        def optional_norm(norm):
            return None if type(norm).__name__ == "Identity" else from_torch(norm)

        return cls(
            node_norm=from_torch(m.node_norm),
            to_qkv=from_torch(m.to_qkv),
            to_g=from_torch(m.to_g),
            to_out_node=from_torch(m.to_out_node),
            q_layer_norm=optional_norm(m.q_layer_norm),
            k_layer_norm=optional_norm(m.k_layer_norm),
            pair_norm=None if m.pair_norm is None else from_torch(m.pair_norm),
            to_bias=None if m.to_bias is None else from_torch(m.to_bias),
            heads=m.heads,
        )

    def __call__(
        self, x: Float[Array, "N D"], pair: Float[Array, "N N P"] | None, mask: Bool[Array, " N"]
    ) -> Float[Array, "N D"]:
        x = self.node_norm(x)
        q, k, v = jnp.split(self.to_qkv(x), 3, axis=-1)
        if self.q_layer_norm is not None:
            q, k = self.q_layer_norm(q), self.k_layer_norm(k)
        q, k, v, g = (einops.rearrange(t, "n (h d) -> n h d", h=self.heads) for t in (q, k, v, self.to_g(x)))
        n = x.shape[0]
        bias = jnp.zeros((self.heads, n, n), x.dtype)
        if self.to_bias is not None:
            bias = einops.rearrange(self.to_bias(self.pair_norm(pair)), "i j h -> h i j")
        pair_mask = mask[:, None] & mask[None, :]
        bias = jnp.where(pair_mask[None], bias, NEG_INF)
        bias = jnp.where(mask[None, :, None], bias, 0.0)  # fully padded query rows attend uniformly
        o = jax.nn.dot_product_attention(q[None], k[None], v[None], bias=bias[None])[0]
        return self.to_out_node(einops.rearrange(jax.nn.sigmoid(g) * o, "n h d -> n (h d)"))


@register_from_torch("rfproteina.nn.modules.pair_bias_attn.MultiHeadBiasedAttentionADALN_MM")
class MultiHeadBiasedAttentionADALN(AbstractFromTorch):
    adaln: AdaptiveLayerNorm
    mha: PairBiasAttention
    scale_output: AdaptiveOutputScale

    def __call__(self, x, pair, cond, mask):
        x = self.adaln(x, cond, mask)
        x = self.mha(x, pair, mask)
        return masked(self.scale_output(x, cond, mask), mask)


@register_from_torch("rfproteina.nn.modules.attn_n_transition.MultiheadAttnAndTransition")
class AttnAndTransition(eqx.Module):
    """Residual pair-biased attention followed by a residual transition (sequential, not parallel)."""

    mhba: MultiHeadBiasedAttentionADALN
    transition: TransitionADALN
    use_attn_pair_bias: bool = eqx.field(static=True)

    @classmethod
    def from_torch(cls, m):
        if m.parallel or not (m.residual_mha and m.residual_transition):
            raise NotImplementedError("only sequential residual attention + transition is ported")
        return cls(mhba=from_torch(m.mhba), transition=from_torch(m.transition), use_attn_pair_bias=m.use_attn_pair_bias)

    def __call__(
        self, x: Float[Array, "N D"], pair: Float[Array, "N N P"] | None, cond: Float[Array, "N C"], mask: Bool[Array, " N"]
    ) -> Float[Array, "N D"]:
        x = masked(x, mask)
        x = masked(self.mhba(x, pair if self.use_attn_pair_bias else None, cond, mask) + x, mask)
        x = masked(self.transition(x, cond, mask) + x, mask)
        return masked(x, mask)


class TriangleMultiplication(eqx.Module):
    """AF2 Algorithms 11 (outgoing) and 12 (incoming)."""

    layer_norm_in: LayerNorm
    layer_norm_out: LayerNorm
    linear_a_p: Linear
    linear_a_g: Linear
    linear_b_p: Linear
    linear_b_g: Linear
    linear_g: Linear
    linear_z: Linear
    outgoing: bool = eqx.field(static=True)

    @classmethod
    def from_torch(cls, m):
        names = ("layer_norm_in", "layer_norm_out", "linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g",
                 "linear_g", "linear_z")
        return cls(**{name: from_torch(getattr(m, name)) for name in names}, outgoing=m._direction == "outgoing")

    def __call__(self, z: Float[Array, "N N P"], pair_mask: Bool[Array, "N N"]) -> Float[Array, "N N P"]:
        z = self.layer_norm_in(z)
        mask = pair_mask[..., None]
        a = self.linear_a_p(z) * jax.nn.sigmoid(self.linear_a_g(z)) * mask
        b = self.linear_b_p(z) * jax.nn.sigmoid(self.linear_b_g(z)) * mask
        x = jnp.einsum("ikc,jkc->ijc", a, b) if self.outgoing else jnp.einsum("kic,kjc->ijc", a, b)
        x = self.linear_z(self.layer_norm_out(x))
        return x * jax.nn.sigmoid(self.linear_g(z))


TORCH_TRI = "rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update"
register_from_torch(f"{TORCH_TRI}.TriangleMultiplicationOutgoing", f"{TORCH_TRI}.TriangleMultiplicationIncoming")(
    TriangleMultiplication
)


@register_from_torch("rfproteina.nn.modules.openfold_modules.pair_transition.PairTransition")
class PairTransition(AbstractFromTorch):
    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear

    def __call__(self, z: Float[Array, "N N P"], pair_mask: Bool[Array, "N N"]) -> Float[Array, "N N P"]:
        return self.linear_2(jax.nn.relu(self.linear_1(self.layer_norm(z)))) * pair_mask[..., None]


@register_from_torch("rfproteina.nn.modules.pair_update.PairReprUpdate")
class PairReprUpdate(eqx.Module):
    """Single-to-pair write (sum or Hadamard), optional triangle multiplications, pair transition."""

    layer_norm_in: LayerNorm
    linear_x: Linear
    transition_out: PairTransition
    tri_mult_out: TriangleMultiplication | None
    tri_mult_in: TriangleMultiplication | None
    hadamard_write: bool = eqx.field(static=True)

    @classmethod
    def from_torch(cls, m):
        return cls(
            layer_norm_in=from_torch(m.layer_norm_in),
            linear_x=from_torch(m.linear_x),
            transition_out=from_torch(m.transition_out),
            tri_mult_out=from_torch(m.tri_mult_out) if m.use_tri_mult else None,
            tri_mult_in=from_torch(m.tri_mult_in) if m.use_tri_mult else None,
            hadamard_write=m.pair_write == "hadamard",
        )

    def __call__(self, x: Float[Array, "N D"], z: Float[Array, "N N P"], mask: Bool[Array, " N"]) -> Float[
        Array, "N N P"
    ]:
        pair_mask = mask[:, None] & mask[None, :]
        p_i, p_j = jnp.split(self.linear_x(self.layer_norm_in(masked(x, mask))), 2, axis=-1)
        z = z + (p_i[None, :, :] * p_j[:, None, :] if self.hadamard_write else p_i[None, :, :] + p_j[:, None, :])
        z = z * pair_mask[..., None]
        if self.tri_mult_out is not None:
            z = (z + self.tri_mult_out(z, pair_mask)) * pair_mask[..., None]
            z = (z + self.tri_mult_in(z, pair_mask)) * pair_mask[..., None]
        return (z + self.transition_out(z, pair_mask)) * pair_mask[..., None]
