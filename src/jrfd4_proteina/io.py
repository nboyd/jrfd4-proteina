"""Torch-free save/load of a converted `RFD4Proteina`."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import equinox as eqx
import jax

from jrfd4_proteina.sampling import RFD4Proteina


def save(model: RFD4Proteina, path: str | os.PathLike) -> None:
    """Write `<path>.eqx` (arrays) and `<path>.skeleton.pkl` (structure, shapes and dtypes)."""
    path = Path(path)
    eqx.tree_serialise_leaves(path.with_suffix(".eqx"), model)
    skeleton = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x, model)
    with open(path.with_suffix(".skeleton.pkl"), "wb") as f:
        pickle.dump(skeleton, f)


def load(path: str | os.PathLike) -> RFD4Proteina:
    path = Path(path)
    with open(path.with_suffix(".skeleton.pkl"), "rb") as f:
        skeleton = pickle.load(f)
    return eqx.tree_deserialise_leaves(path.with_suffix(".eqx"), skeleton)
