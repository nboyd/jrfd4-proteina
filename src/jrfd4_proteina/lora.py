"""LoRA adapters for the flow network, as a pytree separate from the model.

    adapters = lora.init(model.flow, key)                # {path: LoRA}; the only trainable parameters
    flow = lora.attach(model.flow, adapters)             # low-rank updates applied in the forward pass
    flow = lora.merge(flow)                              # folded into plain weights, for sampling

The reference fine-tunes with PEFT adapters on the attention projections `to_qkv` and `to_out_node` of every
trunk layer (rank 32, alpha 64). Trunk layers are stacked along leading axes, and so are their adapters.
"""

from __future__ import annotations

import math
from operator import attrgetter

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jrfd4_proteina.layers import Linear, LoRALinear

TARGETS = ("to_qkv", "to_out_node")


class LoRA(eqx.Module):
    """The update `(alpha / rank) * b @ a` to one (stack of) weight matrices."""

    a: Float[Array, "*layers R In"]
    b: Float[Array, "*layers Out R"]
    alpha: float = eqx.field(static=True)

    @property
    def scale(self) -> float:
        return self.alpha / self.a.shape[-2]


Adapters = dict[str, LoRA]  # keyed by the dotted attribute path of the adapted `Linear`, e.g. "trunk.attn.mhba.mha.to_qkv"


def target_paths(module: eqx.Module, targets: tuple[str, ...] = TARGETS) -> list[str]:
    """Dotted paths of every `Linear` in `module` held in a field named in `targets`."""
    paths = []
    for path, leaf in jax.tree.leaves_with_path(module, is_leaf=lambda x: isinstance(x, Linear)):
        if isinstance(leaf, Linear) and getattr(path[-1], "name", None) in targets:
            if not all(isinstance(key, jax.tree_util.GetAttrKey) for key in path):
                raise NotImplementedError(f"adapter target inside a container: {jax.tree_util.keystr(path)}")
            paths.append(".".join(key.name for key in path))
    return paths


def init(
    module: eqx.Module, key: PRNGKeyArray, *, rank: int = 32, alpha: float = 64.0, targets: tuple[str, ...] = TARGETS
) -> Adapters:
    """Fresh adapters for `module` (PEFT's initialisation: `a` uniform within 1/sqrt(fan_in), `b` zero, so the
    adapted module starts out equal to `module`)."""
    paths = target_paths(module, targets)
    if not paths:
        raise ValueError(f"no Linear named {targets} to adapt")
    adapters = {}
    for path, k in zip(paths, jax.random.split(key, len(paths)), strict=True):
        *layers, n_out, n_in = attrgetter(path)(module).weight.shape
        bound = 1.0 / math.sqrt(n_in)
        a = jax.random.uniform(k, (*layers, rank, n_in), jnp.float32, -bound, bound)
        adapters[path] = LoRA(a, jnp.zeros((*layers, n_out, rank), jnp.float32), alpha)
    return adapters


def _replace(module: eqx.Module, paths: list[str], replacements: list[Linear]) -> eqx.Module:
    return eqx.tree_at(lambda m: [attrgetter(path)(m) for path in paths], module, replacements)


def attach(module: eqx.Module, adapters: Adapters) -> eqx.Module:
    """`module` with each adapted `Linear` replaced by a `LoRALinear`. Free: it only rearranges the pytree, so
    call it inside the loss to differentiate with respect to `adapters`."""
    paths = list(adapters)
    linears = []
    for path in paths:
        base, lora = attrgetter(path)(module), adapters[path]
        if isinstance(base, LoRALinear):
            raise ValueError(f"{path} already has an adapter")  # noqa: TRY004
        linears.append(LoRALinear(base.weight, base.bias, a=lora.a, b=lora.b, scale=lora.scale))
    return _replace(module, paths, linears)


def split(module: eqx.Module) -> tuple[eqx.Module, Adapters]:
    """The inverse of `attach`: the base module and its adapters (e.g. of a converted PEFT model)."""
    paths = [
        ".".join(key.name for key in path)
        for path, leaf in jax.tree.leaves_with_path(module, is_leaf=lambda x: isinstance(x, Linear))
        if isinstance(leaf, LoRALinear)
    ]
    linears = [attrgetter(path)(module) for path in paths]
    adapters = {path: LoRA(m.a, m.b, m.scale * m.a.shape[-2]) for path, m in zip(paths, linears, strict=True)}
    return _replace(module, paths, [Linear(m.weight, m.bias) for m in linears]), adapters


def merge(module: eqx.Module, adapters: Adapters | None = None) -> eqx.Module:
    """Fold adapters (those attached to `module`, or `adapters`) into plain weights: same function, no overhead.

    Fold into float32 weights (before `RFD4Proteina.bfloat16()`); bf16 weights would round the update away.
    """
    base, attached = split(module)
    if adapters is not None and attached:
        raise ValueError("module already has adapters attached")
    adapters = attached if adapters is None else adapters
    paths = list(adapters)
    linears = []
    for path in paths:
        linear, lora = attrgetter(path)(base), adapters[path]
        if linear.weight.dtype != jnp.float32:
            raise ValueError(f"{path}: merge into float32 weights, not {linear.weight.dtype}")
        linears.append(Linear(linear.weight + lora.scale * lora.b @ lora.a, linear.bias))
    return _replace(base, paths, linears)
