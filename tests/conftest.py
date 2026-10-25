"""Shared fixtures for the opt-in `reference` tests (PyTorch reference on CPU, fp32)."""

import os
from pathlib import Path

import pytest

BUNDLE = "nvidia/RFD4-Proteina"
REVISION = "48377e45eb25a4fce8d2aff56c1a659b726d2106"


@pytest.fixture(scope="session")
def jax_cpu_highest():
    import jax

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_default_matmul_precision", "highest")
    return jax


@pytest.fixture(scope="session")
def reference(jax_cpu_highest):
    """(torch model, jax model, torch batch, jax batch) for the shipped 100-residue monomer benchmark."""
    import rfproteina
    import torch

    from jrfd4_proteina import convert

    torch.set_float32_matmul_precision("highest")
    torch_model = convert.load_reference(BUNDLE, revision=REVISION).eval()
    jax_model = convert.convert_model(torch_model)
    cif = Path(rfproteina.__file__).parent / "data/benchmarks/monomer/monomer-short.cif.zst"
    os.environ.setdefault("DATA_PATH", str(Path.home() / ".cache" / "jrfd4_proteina"))
    torch_batch = _reference_batch(str(cif))
    return torch_model, jax_model, torch_batch, convert.batch_from_reference(torch_batch)[0]


@pytest.fixture(scope="session")
def pdl1_reference(reference):
    """As `reference`, for the reference's PD-L1 binder example (two molecules, coordinate group, hotspots)."""
    import random

    from jrfd4_proteina import convert

    torch_model, jax_model, _, _ = reference
    random.seed(0)  # the binder length is sampled from the contig with `random`
    torch_batch = _reference_batch(str(Path(__file__).parent / "data" / "pdl1_binder.yaml"))
    return torch_model, jax_model, torch_batch, convert.batch_from_reference(torch_batch)[0]


def _reference_batch(inputs: str) -> dict:
    from hydra import compose, initialize_config_dir
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.generate import get_dataset_from_cfg

    from jrfd4_proteina.convert import reference_config_dir

    with initialize_config_dir(config_dir=str(reference_config_dir()), version_base="1.3"):
        cfg = compose("inference", overrides=[f"inputs={inputs}", "num_replicates=1", "~metrics"])
    batch = collate_batch(
        [get_dataset_from_cfg(cfg, 0)[0]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
        fixed_dimensions=None,
    )
    batch["coords_nm"] = batch["coords"] / 10.0
    batch["coord_mask"] = batch["atom37_mask"]
    return batch
