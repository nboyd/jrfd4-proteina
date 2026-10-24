"""`from_torch` conversion machinery and scan helpers.

Importing this module does not import torch: torch modules are matched by dotted
type path, resolved only by `jrfd4_proteina.convert`.
"""

from __future__ import annotations

import dataclasses
from functools import singledispatch
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

# Dotted torch type path -> converter (a JAX class with `from_torch`, or a callable).
TORCH_REGISTRY: dict[str, Any] = {}


def register_from_torch(*torch_paths: str):
    """Class decorator: convert torch modules of the given dotted type paths with `cls.from_torch`."""

    def decorator(cls):
        for path in torch_paths:
            TORCH_REGISTRY[path] = cls
        return cls

    return decorator


@singledispatch
def from_torch(obj):
    path = f"{type(obj).__module__}.{type(obj).__qualname__}"
    if path in TORCH_REGISTRY:
        target = TORCH_REGISTRY[path]
        return target.from_torch(obj) if hasattr(target, "from_torch") else target(obj)
    raise TypeError(f"no from_torch converter registered for {path}")


for _type in (int, float, bool, str, type(None)):
    from_torch.register(_type, lambda x: x)
from_torch.register(tuple, lambda t: tuple(from_torch(v) for v in t))
from_torch.register(list, lambda t: [from_torch(v) for v in t])


class AbstractFromTorch(eqx.Module):
    """Convert a torch module by matching dataclass fields to its children, parameters and buffers."""

    @classmethod
    def from_torch(cls, module) -> AbstractFromTorch:
        fields = {f.name: f for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, child in module.named_children():
            if name in fields:
                kwargs[name] = from_torch(child)
        for name, tensor in [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]:
            if name in fields:
                kwargs[name] = from_torch(tensor)
        for name, field in fields.items():
            if name in kwargs:
                continue
            if hasattr(module, name) and isinstance(getattr(module, name), (int, float, bool, str, type(None))):
                kwargs[name] = getattr(module, name)
            elif field.default is not dataclasses.MISSING:
                kwargs[name] = field.default
            else:
                raise ValueError(f"{cls.__name__}: no value for field {name!r} on {type(module).__name__}")
        return cls(**kwargs)


def stack_modules(modules: list[eqx.Module]) -> tuple[eqx.Module, eqx.Module]:
    """Stack identically structured modules: (array leaves stacked on a leading axis, shared static part)."""
    arrays = [eqx.filter(m, eqx.is_array) for m in modules]
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *arrays)
    _, static = eqx.partition(modules[0], eqx.is_array)
    return stacked, static
