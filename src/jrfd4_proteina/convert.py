"""Conversion from the PyTorch reference (`rfd4-proteina`). The only module that imports torch.

    from jrfd4_proteina import convert
    model = convert.load("nvidia/RFD4-Proteina")          # Hugging Face repo or local bundle
    batch = convert.featurize({"design": spec})[0]        # reference featurization of a condition spec
"""

from __future__ import annotations

import copy
import importlib
import itertools
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

import jrfd4_proteina.features
import jrfd4_proteina.layers
import jrfd4_proteina.model  # noqa: F401
from jrfd4_proteina.backend import REGISTRY, from_torch
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.sampling import NoiseConfig, RFD4Proteina
from jrfd4_proteina.training import Example, Targets


def _resolve(path: str) -> type:
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


for _path, _convert in REGISTRY:
    from_torch.register(_resolve(_path), _convert)
from_torch.register(torch.nn.Identity, lambda m: None)


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
    """Convert a loaded `rfproteina.proteina.Proteina` (flow network + AE encoder and decoder).

    PEFT LoRA adapters on the denoiser convert to `LoRALinear`s; `lora.split(model.flow)` separates them.
    """
    proteina = proteina.eval().float()
    fm = proteina.cfg_exp.product_flowmatcher
    if list(fm) != ["bb_ca", "local_latents"]:
        raise NotImplementedError(f"modalities {list(fm)} are not supported")
    return RFD4Proteina(
        flow=from_torch(proteina.nn),
        decoder=from_torch(proteina.autoencoder.decoder),
        encoder=from_torch(proteina.autoencoder.encoder),
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


@contextmanager
def _seeded(seed: int | None) -> Iterator[None]:
    """Seed the global RNGs the reference pipeline draws from, restoring their states afterwards."""
    if seed is None:
        yield
        return
    states = random.getstate(), np.random.get_state()
    random.seed(seed)  # contig lengths are sampled with `random`, free-pose internal coordinates with torch
    np.random.seed(seed)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])


def featurize(spec: Mapping[str, Mapping], *, seed: int | None = None) -> list[Batch]:
    """Featurize a condition spec with the reference data pipeline, one `Batch` per example.

    `spec` is the reference's condition-spec YAML as a dict, `{example_id: {"input": ..., "contig": ...,
    "conditions": ...}}`; relative `input` paths resolve against the working directory. Contig length ranges
    are sampled anew on each call; `seed` makes that reproducible without touching the global RNG states.
    """
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.inference_utils.datasets import ConditionSpecDataset

    os.environ.setdefault("DATA_PATH", str(Path.home() / ".cache" / "jrfd4_proteina"))
    batches = []
    with _seeded(seed):
        dataset = ConditionSpecDataset(
            data=copy.deepcopy(dict(spec)), transform=build_design_validation_pipeline(), name="jrfd4-proteina"
        )
        for i in range(len(dataset)):
            collated = collate_batch(
                [dataset[i]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample", fixed_dimensions=None
            )
            batches.extend(batch_from_reference(collated))
    return batches


def training_examples(batch: dict) -> Example:
    """A reference collated training batch as one stacked `Example` (leading axis: examples), on the host."""
    arrays = {k: v.detach().cpu().numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    return Example(Batch.from_reference(arrays), Targets.from_reference(arrays))


def train_loader(overrides: Sequence[str] = ()) -> Iterator[Example]:
    """Stacked `Example`s from the reference training dataloader, without end.

    The loader is the reference's own: its `training_proteina` Hydra config composed with `overrides`
    (`experiment=...`, `dataloaders.batch_size=...`, `dataloaders.num_workers=...`, `_max_tokens=...`), so
    datasets, design tasks, cropping and padding to a fixed token count are exactly as it trains with. Each
    example appears once: noise candidates (`exploration_k`) are drawn in JAX.
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from rfproteina.utils.datasets import assemble_distributed_loader, recursively_instantiate_datasets_and_samplers
    from rfproteina.utils.hydra import register_resolvers

    os.environ.setdefault("DATA_PATH", str(Path.home() / ".cache" / "jrfd4_proteina"))
    register_resolvers()
    with initialize_config_dir(config_dir=str(reference_config_dir()), version_base="1.3"):
        cfg = compose("training_proteina", overrides=[*overrides, "training.exploration_k=1"])
    built = recursively_instantiate_datasets_and_samplers(cfg.datasets.train)
    loader_cfg = hydra.utils.instantiate(cfg.dataloaders.train)
    batches_per_epoch = cfg.opt.steps_per_epoch * cfg.opt.accumulate_grad_batches
    loader = assemble_distributed_loader(
        dataset=built["dataset"],
        sampler=built["sampler"],
        rank=0,
        world_size=1,
        n_examples_per_epoch=batches_per_epoch * loader_cfg["dataloader_params"]["batch_size"],
        loader_cfg=loader_cfg,
    )
    for epoch in itertools.count():
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)  # reshuffle
        for batch in loader:
            yield training_examples(batch)
