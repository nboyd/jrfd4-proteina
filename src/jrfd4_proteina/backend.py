"""`from_torch` conversion machinery.

Importing this module does not import torch: JAX classes register the dotted type paths of the torch
classes they mirror, and `jrfd4_proteina.convert` (the only torch importer) resolves those paths and
registers them with `from_torch`.
"""

from __future__ import annotations

import dataclasses
from functools import singledispatch
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

# (dotted torch type path, converter) pairs, registered with `from_torch` by `jrfd4_proteina.convert`.
REGISTRY: list[tuple[str, Any]] = []


def register_from_torch(*torch_paths: str):
    """Class decorator: convert torch modules of the given dotted type paths with `cls.from_torch`."""

    def wrap(cls):
        REGISTRY.extend((path, cls.from_torch) for path in torch_paths)
        return cls

    return wrap


@singledispatch
def from_torch(obj: Any) -> Any:
    raise TypeError(f"no from_torch conversion for {type(obj).__module__}.{type(obj).__qualname__}")


for _scalar in (int, float, bool, str, type(None)):
    from_torch.register(_scalar, lambda x: x)


class AbstractFromTorch(eqx.Module):
    """Convert a torch module by matching dataclass fields to its attributes (children, parameters, buffers,
    scalars). Fields the module lacks fall back to their defaults; torch parameters, and children holding
    parameters, with no matching field are an error, so no weights are silently dropped."""

    @classmethod
    def from_torch(cls, module: Any) -> AbstractFromTorch:
        fields = dataclasses.fields(cls)
        names = {f.name for f in fields}
        weighted = [n for n, child in module.named_children() if next(child.parameters(), None) is not None]
        weighted += [n for n, _ in module.named_parameters(recurse=False)]
        if unmatched := [n for n in weighted if n not in names]:
            raise ValueError(f"{cls.__name__}: unmatched torch children/parameters {unmatched}")
        kwargs: dict[str, Any] = {}
        for f in fields:
            if hasattr(module, f.name):
                kwargs[f.name] = from_torch(getattr(module, f.name))
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            else:
                raise ValueError(f"{cls.__name__}: no value for field {f.name!r} on {type(module).__name__}")
        return cls(**kwargs)


def stack_modules(modules: list[eqx.Module]) -> eqx.Module:
    """Identically structured modules as one module whose leaves gain a leading layer axis (to scan over).
    `jax.tree.map` rejects differing static structure."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *modules)
