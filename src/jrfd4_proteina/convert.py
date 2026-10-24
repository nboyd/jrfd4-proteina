"""Conversion from the PyTorch reference (`rfd4-proteina`). The only module that imports torch.

    from jrfd4_proteina import convert
    model = convert.load("nvidia/RFD4-Proteina")          # Hugging Face repo or local bundle
    batch = convert.featurize("design.yaml")[0]           # reference featurization of a design spec
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import torch

import jrfd4_proteina.features
import jrfd4_proteina.layers
import jrfd4_proteina.model  # noqa: F401
from jrfd4_proteina.backend import from_torch
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.sampling import NoiseConfig, RFD4Proteina


@from_torch.register(torch.Tensor)
def _tensor(t: torch.Tensor):
    t = t.detach().cpu()
    return jnp.array((t.float() if t.dtype == torch.bfloat16 else t).numpy())


def _noise_config(cfg) -> NoiseConfig:
    mode = cfg.get("stochastic_centering_mode", "none")
    if mode not in ("none", "x_0"):
        raise NotImplementedError(f"stochastic centering mode {mode!r} is not ported")
    if mode == "x_0" and (
        cfg.get("stochastic_centering_scope") != "per_group" or list(cfg.get("stochastic_centering_group_keys")) !=
        ["molecule_iid"]
    ):
        raise NotImplementedError("only per-molecule stochastic centering is ported")
    return NoiseConfig(
        zero_com=bool(cfg.get("zero_com_noise", False)),
        centering_scale=float(cfg["stochastic_centering_scale"]) if mode == "x_0" else None,
    )


def convert_model(proteina) -> RFD4Proteina:
    """Convert a loaded `rfproteina.proteina.Proteina` (flow network + AE decoder)."""
    proteina = proteina.eval().float()
    fm = proteina.cfg_exp.product_flowmatcher
    if list(fm) != ["bb_ca", "local_latents"]:
        raise NotImplementedError(f"modalities {list(fm)} are not supported")
    return RFD4Proteina(
        flow=from_torch(proteina.nn),
        decoder=from_torch(proteina.autoencoder.decoder),
        noise_bb_ca=_noise_config(fm.bb_ca),
        noise_local_latents=_noise_config(fm.local_latents),
        latent_dim=int(fm.local_latents.dim),
    )


def load_reference(bundle: str | os.PathLike, revision: str | None = None):
    """Load the torch reference model from a Hugging Face repo id or local bundle directory."""
    from rfproteina.proteina import Proteina

    return Proteina.from_pretrained(str(bundle), revision=revision)


def load(bundle: str | os.PathLike, revision: str | None = None) -> RFD4Proteina:
    return convert_model(load_reference(bundle, revision))


def batch_from_reference(batch: dict) -> list[Batch]:
    """Split a reference collated batch (dict of [b, ...] tensors) into per-example `Batch`es."""
    arrays = {k: v.detach().cpu().numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    n_examples = arrays["padding_mask"].shape[0]
    return [Batch.from_reference({k: v[i] for k, v in arrays.items()}) for i in range(n_examples)]


def reference_config_dir() -> Path:
    import rfproteina

    return Path(rfproteina.__file__).parent / "configs"


def featurize(inputs: str | os.PathLike, overrides: Sequence[str] = ()) -> list[Batch]:
    """Featurize a design spec (YAML/JSON/annotated CIF) with the reference data pipeline."""
    from hydra import compose, initialize_config_dir
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.generate import get_dataset_from_cfg

    os.environ.setdefault("DATA_PATH", str(Path.home() / ".cache" / "jrfd4_proteina"))
    with initialize_config_dir(config_dir=str(reference_config_dir()), version_base="1.3"):
        cfg = compose("inference", overrides=[f"inputs={inputs}", "num_replicates=1", "~metrics", *overrides])
    dataset = get_dataset_from_cfg(cfg, 0)
    batches = []
    for i in range(len(dataset)):
        collated = collate_batch(
            [dataset[i]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample", fixed_dimensions=None
        )
        batches.extend(batch_from_reference(collated))
    return batches

