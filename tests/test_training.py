"""LoRA fine-tuning: torch-free tests (run by default) and parity against the PyTorch reference (`-m reference`)."""

import copy
import os
import random
import subprocess
import sys
import tarfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jrfd4_proteina import lora, optim, training
from jrfd4_proteina.layers import Linear


def rel(ours, theirs) -> float:
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    return float(np.abs(ours - theirs).max() / max(np.abs(theirs).max(), 1e-12))


# --------------------------------------------------------------------------- torch-free


def test_import_does_not_pull_in_torch():
    code = "import sys, jrfd4_proteina.training, jrfd4_proteina.lora; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


class Attention(eqx.Module):
    to_qkv: Linear
    to_out_node: Linear
    to_g: Linear


class Stack(eqx.Module):
    layers: Attention  # leaves stacked as [3, ...]

    def __call__(self, x):
        def layer(x, m):
            return (m.to_out_node(m.to_qkv(x)) + m.to_g(x)).astype(jnp.float32), None

        return jax.lax.scan(layer, x, self.layers)[0]


def _stack(key) -> Stack:
    kq, ko, kg = jax.random.split(key, 3)
    weight = lambda k, n_out, n_in: jax.random.normal(k, (3, n_out, n_in)) / n_in**0.5
    return Stack(Attention(Linear(weight(kq, 12, 6)), Linear(weight(ko, 6, 12), jnp.ones((3, 6))), Linear(weight(kg, 6, 6))))


def test_lora_attach_merge_and_split_agree():
    key = jax.random.key(0)
    net, x = _stack(key), jax.random.normal(key, (5, 6))
    adapters = lora.init(net, key, rank=2, alpha=8.0)
    assert set(adapters) == {"layers.to_qkv", "layers.to_out_node"}  # `to_g` is not a target
    assert adapters["layers.to_qkv"].a.shape == (3, 2, 6) and adapters["layers.to_qkv"].b.shape == (3, 12, 2)
    assert adapters["layers.to_qkv"].scale == 4.0
    np.testing.assert_allclose(lora.attach(net, adapters)(x), net(x), rtol=1e-6)  # b starts at zero

    adapters = jax.tree.map(lambda leaf: leaf + 0.3 * jax.random.normal(key, leaf.shape), adapters)
    attached = lora.attach(net, adapters)
    assert rel(attached(x), net(x)) > 0.1
    with jax.default_matmul_precision("highest"):  # the GPU default (TF32) agrees to ~1e-3 only
        np.testing.assert_allclose(lora.merge(attached)(x), attached(x), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(lora.merge(net, adapters)(x), attached(x), rtol=1e-5, atol=1e-4)

    base, recovered = lora.split(attached)
    assert jax.tree.structure(base) == jax.tree.structure(net)
    assert all(eqx.tree_equal(recovered[path], adapters[path]) for path in adapters)
    with pytest.raises(ValueError):
        lora.attach(attached, adapters)


def test_lora_update_survives_bfloat16_base_weights():
    """The update is added after the matmuls: folded into bf16 weights it would round away."""
    from jrfd4_proteina.layers import matmul_dtype

    key = jax.random.key(1)
    net, x = _stack(key), jax.random.normal(key, (5, 6))
    adapters = lora.init(net, key, rank=2, alpha=1e-3)
    adapters = jax.tree.map(lambda leaf: leaf + jax.random.normal(key, leaf.shape), adapters)
    attached = lora.attach(matmul_dtype(net, jnp.bfloat16), adapters)
    grads = jax.grad(lambda a: lora.attach(matmul_dtype(net, jnp.bfloat16), a)(x).astype(jnp.float32).sum())(adapters)
    assert attached.layers.to_qkv.a.dtype == jnp.float32 and attached.layers.to_qkv.weight.dtype == jnp.bfloat16
    assert all(g.dtype == jnp.float32 and jnp.abs(g).max() > 0 for g in jax.tree.leaves(grads))
    with pytest.raises(ValueError):
        lora.merge(attached)


def test_muon_orthogonalizes_each_stacked_matrix():
    g = jax.random.normal(jax.random.key(0), (4, 3, 16, 5)) * jnp.logspace(-4, 2, 4)[:, None, None, None]
    tx = optim.scale_by_muon(momentum=0.0)
    update, _ = tx.update({"w": g}, tx.init({"w": g}))
    singular = np.linalg.svd(np.asarray(update["w"]) / (0.2 * 16**0.5), compute_uv=False)
    assert singular.shape == (4, 3, 5) and singular.min() > 0.3 and singular.max() < 1.6
    aligned = jnp.sum(update["w"] * g, axis=(-2, -1))
    assert bool((aligned > 0).all())  # a descent direction, whatever the gradient's scale
    with pytest.raises(ValueError):
        tx.init({"bias": jnp.zeros(3)})


def test_flow_matching_loss_masks_and_weights():
    key = jax.random.key(0)
    clean, noise = jax.random.normal(key, (6, 3)), jax.random.normal(jax.random.key(1), (6, 3))
    supervised = jnp.array([True, True, True, False, False, False])
    weight = jnp.array([1.0, 2.0, 0.0, 5.0, 5.0, 5.0])
    t = jnp.float32(0.5)
    assert training.flow_matching_loss(clean, clean, t, supervised, weight) == 0.0
    loss = training.flow_matching_loss(clean + noise, clean, t, supervised, weight)
    expected = ((noise[0] ** 2).sum() + 2.0 * (noise[1] ** 2).sum()) / 3 / (0.25 + 1e-5)
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
    assert training.flow_matching_loss(clean + noise, clean, t, jnp.zeros(6, bool), weight) == 0.0


def test_signed_volume_loss_penalises_the_mirror_image():
    centre = jnp.zeros(3)
    vertices = jnp.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, -1.0, -1.0]])
    coords = jnp.concatenate([centre[None], vertices]) * 0.15
    tetrahedra = jnp.full((5, 4), -1).at[0].set(jnp.array([1, 2, 3, 4]))
    supervised = jnp.ones(5, bool)
    assert training.signed_volume_loss(coords, coords, tetrahedra, supervised) == 0.0
    assert training.signed_volume_loss(coords * jnp.array([1.0, 1.0, -1.0]), coords, tetrahedra, supervised) > 0.1
    assert training.signed_volume_loss(-coords, coords, tetrahedra, supervised.at[3].set(False)) == 0.0


def test_time_and_recycle_samplers_stay_in_range():
    config = training.release_training_config()
    keys = jax.random.split(jax.random.key(0), 2000)
    for dist in (config.t_bb_ca, config.t_local_latents):
        t = jax.vmap(dist.sample)(keys)
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    recycling = jax.vmap(config.recycle.sample)(keys)
    assert int(recycling.depth.min()) == 1 and int(recycling.depth.max()) <= 9
    assert 0.4 < float(recycling.condition_on_remaining.mean()) < 0.6
    assert float(jax.vmap(config.t_bb_ca.sample)(keys).mean()) > float(jax.vmap(config.t_local_latents.sample)(keys).mean())


# --------------------------------------------------------------------------- reference parity


@pytest.fixture(scope="module")
def training_batch(reference):
    """A reference collated training batch of three examples from its tutorial dataset, padded to 64 tokens:
    a plain complex, one with unindexed motif tokens (insertion targets, context, loss weights) and one with
    tokens that carry no latent."""
    import rfproteina.utils.hydra
    import torch
    from conftest import BUNDLE, REVISION
    from huggingface_hub import hf_hub_download
    from hydra import compose, initialize_config_dir
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.utils.datasets import recursively_instantiate_datasets_and_samplers

    from jrfd4_proteina.convert import reference_config_dir

    data = Path(os.environ["DATA_PATH"])
    if not (data / "test_dataset" / "test_systems.parquet").exists():
        archive = hf_hub_download(BUNDLE, "test_dataset/test_dataset.tar.gz", revision=REVISION, local_dir=data)
        with tarfile.open(archive) as tar:
            tar.extractall(data / "test_dataset", filter="data")
    rfproteina.utils.hydra.register_resolvers()
    with initialize_config_dir(config_dir=str(reference_config_dir()), version_base="1.3"):
        cfg = compose("training_proteina", overrides=["experiment=tutorial/finetune_test_dataset"])
    dataset = recursively_instantiate_datasets_and_samplers(cfg.datasets.train)["dataset"]
    samples = []
    for index in (4, 8, 12):
        random.seed(index)  # design tasks are sampled
        np.random.seed(index)
        torch.manual_seed(index)
        samples.append(dataset[index])
    batch = collate_batch(
        samples, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample", fixed_dimensions={"I": 64},
        keep_atom_array=False,
    )
    assert (batch["insertion_source_idx"] >= 0).any() and batch["is_in_context_conditioning"].any()
    assert not batch["supports_local_latent"][batch["padding_mask"]].all() and (batch["loss_weight"] != 1).any()
    return batch


@pytest.fixture(scope="module")
def lora_reference(reference):
    """A copy of the torch model with PEFT adapters (random, so the update is not zero) and its conversion."""
    import torch
    from omegaconf import OmegaConf
    from rfproteina.utils.lora import add_lora_adapter

    from jrfd4_proteina import convert

    torch_model = copy.deepcopy(reference[0])
    add_lora_adapter(torch_model, OmegaConf.create(
        {"r": 8, "lora_alpha": 24.0, "lora_dropout": 0.0, "train_bias": "none",
         "target_modules": ["to_qkv", "to_out_node"]}
    ))
    torch.manual_seed(0)
    for name, parameter in torch_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.02)
    torch.backends.cuda.enable_math_sdp(True)  # the only SDPA kernel with a CPU backward
    return torch_model.eval(), convert.convert_model(torch_model)


@pytest.mark.reference
@pytest.mark.parametrize(("n_recycle", "condition"), [(1, True), (3, True), (3, False)])
def test_losses_and_adapter_gradients_match_reference(lora_reference, training_batch, n_recycle, condition):
    """Per example: corrupt with the reference, then compare every loss term and the gradients of the mean
    objective with respect to all adapters."""
    import torch

    from jrfd4_proteina import convert
    from jrfd4_proteina.sampling import State

    torch_model, converted = lora_reference
    flow, adapters = lora.split(converted.flow)
    assert set(adapters) == set(lora.target_paths(flow)) and adapters["trunk.attn.mhba.mha.to_qkv"].scale == 3.0
    examples = convert.training_examples(training_batch)
    config = training.release_training_config()
    n_examples = training_batch["padding_mask"].shape[0]

    torch_model.zero_grad()
    reference_terms, corruptions = [], []
    randperm = torch.randperm
    for i in range(n_examples):
        b = {k: v[i : i + 1] if isinstance(v, torch.Tensor) else v for k, v in training_batch.items()}
        b["coords_nm"], b["coord_mask"] = b["coords"] / 10.0, b["atom37_mask"]
        b["condition_on_recycle_remaining"] = condition
        t = {"bb_ca": torch.tensor([0.3 + 0.2 * i]), "local_latents": torch.tensor([0.6 - 0.2 * i])}
        torch.manual_seed(10 + i)
        with torch.no_grad():
            b = torch_model.fm.corrupt_batch(torch_model.add_clean_samples(b), t=t)
        torch.randperm = lambda n, **kwargs: torch.arange(n)  # coordinate groups keep their order
        try:
            out = torch_model.call_nn(b, n_recycle=n_recycle)
        finally:
            torch.randperm = randperm
        terms = torch_model.fm.compute_loss(b, out)
        (sum(v.mean() for v in torch_model.fm.optimization_terms(terms).values()) / n_examples).backward()
        reference_terms.append({k: v.item() for k, v in terms.items()})
        corruptions.append(training.Corruption(
            State(b["x_1"]["bb_ca"][0].numpy(), b["x_1"]["local_latents"][0].numpy()),
            State(b["x_t"]["bb_ca"][0].numpy(), b["x_t"]["local_latents"][0].numpy()),
            np.array([t["bb_ca"][0], t["local_latents"][0]], np.float32),
            np.arange(examples.batch.coord_group[0].size, dtype=np.float32),
        ))
    corruptions = jax.tree.map(lambda *leaves: np.stack(leaves), *corruptions)
    recycling = training.Recycling(jnp.asarray(n_recycle), jnp.asarray(condition))

    @eqx.filter_jit
    def value_and_grad(adapters, flow, examples, corruptions, recycling):
        def objective(adapters):
            attached = lora.attach(flow, adapters)
            score = lambda example, c: training.denoising_losses(attached, example, c, recycling, config)
            terms = jax.vmap(score)(examples, corruptions)
            return terms.total.mean(), terms

        return eqx.filter_value_and_grad(objective, has_aux=True)(adapters)

    (_, terms), grads = value_and_grad(adapters, flow, examples, corruptions, recycling)
    names = {"bb_ca": "bb_ca", "local_latents": "local_latents", "sequence": "sequence_prediction",
             "insertion": "insertion_prediction", "signed_volume": "signed_volume",
             "sequence_accuracy": "sequence_prediction_accuracy_justlog"}
    for ours, theirs in names.items():
        expected = [r[theirs] for r in reference_terms]
        np.testing.assert_allclose(getattr(terms, ours), expected, rtol=2e-5, atol=1e-6, err_msg=ours)
    assert max(r["insertion_prediction"] for r in reference_terms) > 0.1

    layers = torch_model.nn.transformer_layers
    for path, grad in grads.items():
        modules = [layer.mhba.mha.get_submodule(path.rsplit(".", 1)[-1]) for layer in layers]
        for ours, factor in ((grad.a, "lora_A"), (grad.b, "lora_B")):
            theirs = np.stack([getattr(m, factor)["default"].weight.grad.numpy() for m in modules])
            assert np.abs(theirs).max() > 1e-3
            assert rel(ours.reshape(theirs.shape), theirs) < 3e-3, (path, factor)


@pytest.mark.reference
def test_merged_adapters_match_attached_adapters(lora_reference, reference):
    from jrfd4_proteina.features import FlowInputs

    flow, batch = lora_reference[1].flow, reference[3]
    n = batch.padding_mask.shape[0]
    x = jax.random.normal(jax.random.key(0), (n, 3 + lora_reference[1].latent_dim)) * 0.7
    inputs = FlowInputs(x[:, :3], x[:, 3:], jnp.float32(0.55), jnp.float32(0.35), jnp.zeros(batch.coord_group.size))
    attached, merged, base = (f(batch, inputs, 2) for f in (flow, lora.merge(flow), lora.split(flow)[0]))
    assert rel(merged.v_bb_ca, attached.v_bb_ca) < 2e-5 and rel(merged.v_local_latents, attached.v_local_latents) < 2e-5
    assert rel(base.v_bb_ca, attached.v_bb_ca) > 1e-2  # the adapters matter


@pytest.mark.reference
def test_signed_volume_loss_matches_reference():
    import torch
    from rfproteina.flow_matching.signed_volume_loss import signed_volume_loss

    rng = np.random.default_rng(0)
    n = 40
    reference_coords = rng.normal(size=(n, 3)).astype(np.float32) * 0.3
    predicted = reference_coords + rng.normal(size=(n, 3)).astype(np.float32) * 0.2
    tetrahedra = np.stack([rng.choice(n, 4, replace=False) for _ in range(n)])
    tetrahedra[rng.random(n) < 0.3] = -1
    supervised = rng.random(n) < 0.9
    expected = signed_volume_loss(*(torch.tensor(x)[None] for x in (predicted, reference_coords, tetrahedra, supervised)))
    assert expected.item() > 1e-3
    ours = training.signed_volume_loss(predicted, reference_coords, tetrahedra, supervised)
    np.testing.assert_allclose(ours, expected[0].numpy(), rtol=1e-5)


@pytest.mark.reference
def test_muon_matches_torch(monkeypatch):
    """Against `torch.optim.Muon` with its Newton-Schulz iteration in float32 (torch runs it in bf16)."""
    import torch

    monkeypatch.setattr(torch.Tensor, "bfloat16", lambda self: self.clone())
    rng = np.random.default_rng(0)
    shapes = [(8, 48), (64, 8), (16, 16)]
    initial = [rng.normal(size=s).astype(np.float32) for s in shapes]
    parameters = [torch.nn.Parameter(torch.tensor(p)) for p in initial]
    reference_optimizer = torch.optim.Muon(parameters, lr=5e-2, adjust_lr_fn="match_rms_adamw")
    params = [jnp.stack([p, p]) for p in initial]  # two layers each, as trunk adapters are stacked
    tx = optim.muon(5e-2)
    state = tx.init(params)
    for _ in range(6):
        grads = [rng.normal(size=s).astype(np.float32) * 10 ** rng.uniform(-3, 1) for s in shapes]
        for parameter, grad in zip(parameters, grads, strict=True):
            parameter.grad = torch.tensor(grad)
        reference_optimizer.step()
        updates, state = tx.update([jnp.stack([g, g]) for g in grads], state, params)
        params = [p + u for p, u in zip(params, updates, strict=True)]
    for ours, theirs, start in zip(params, parameters, initial, strict=True):
        assert np.abs(theirs.detach().numpy() - start).max() > 0.1
        np.testing.assert_allclose(ours, np.stack([theirs.detach().numpy()] * 2), atol=5e-6)
