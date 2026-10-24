"""Parity against the PyTorch reference (opt-in: `uv run --extra convert pytest -m reference`).

Tolerances are relative to the largest reference value. Two documented fp32 floors apply:
  * the decoder's attention logits reach ~1e3, so its full output agrees to ~2e-5 (torch and this
    port are equally far, ~9e-6, from a float64 evaluation);
  * the release (TSR) sampler amplifies 1e-6 perturbations ~1e4x, so end-to-end trajectories are
    compared under the plain-ODE sampler; the TSR path is checked step-for-step via its noise use.
"""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.reference


def rel(ours, theirs) -> float:
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    return float(np.abs(ours - theirs).max() / max(np.abs(theirs).max(), 1e-12))


def _sampling_args(mode: str) -> dict:
    from omegaconf import OmegaConf

    from jrfd4_proteina.convert import reference_config_dir

    args = OmegaConf.to_container(OmegaConf.load(reference_config_dir() / "generation/inference_base.yaml").model)
    for modality in args.values():
        modality["simulation_step_params"]["sampling_mode"] = mode
    return args


def _predict(torch_model):
    def predict(batch, mode, recycle_depth):
        assert mode == "full"
        return torch_model.call_nn({**batch, "condition_on_recycle_remaining": True}, n_recycle=recycle_depth)

    return predict


def _mid_trajectory_inputs(torch_model, torch_batch):
    import torch
    from rfproteina.flow_matching.product_space_flow_matcher import extract_groups
    from rfproteina.model_masks import compute_flow_emit_masks

    torch.manual_seed(0)
    x = torch_model.fm.sample_noise(
        torch_batch["padding_mask"].shape[1], shape=(1,), emit_masks=compute_flow_emit_masks(torch_batch),
        training=False, coord_cond_mask=torch_batch.get("coord_cond_mask"), groups=extract_groups(torch_batch),
    )
    return {k: v * 0.7 for k, v in x.items()}, {"bb_ca": torch.tensor([0.55]), "local_latents": torch.tensor([0.35])}


def test_schedules_match_reference():
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher

    from jrfd4_proteina.sampling import _schedule, release_sampling_config

    fm = ProductSpaceFlowMatcher.__new__(ProductSpaceFlowMatcher)
    fm.data_modes = ["bb_ca", "local_latents"]
    for nsteps in (24, 60):
        ts, gt = ProductSpaceFlowMatcher.sample_schedule(fm, nsteps, _sampling_args("vf_tsr"))
        ours = _schedule(dataclasses.replace(release_sampling_config(), nsteps=nsteps))
        for m, name in enumerate(fm.data_modes):
            np.testing.assert_allclose(np.append(ours.t[:, m], ours.t_next[-1, m]), ts[name].numpy(), rtol=0, atol=3e-7)
            assert rel(ours.g[:, m], gt[name].numpy()) < 1e-6


def test_flow_forward_parity(reference):
    import jax.numpy as jnp
    import torch

    from jrfd4_proteina.features import FlowInputs

    torch_model, model, torch_batch, batch = reference
    x, t = _mid_trajectory_inputs(torch_model, torch_batch)
    with torch.no_grad():
        ref = _predict(torch_model)({**torch_batch, "x_t": x, "t": t}, "full", 2)
    inputs = FlowInputs(x["bb_ca"][0].numpy(), x["local_latents"][0].numpy(), jnp.float32(0.55), jnp.float32(0.35),
                        jnp.zeros(batch.coord_group.size))
    out = model.flow(batch, inputs, 2)
    assert rel(out.v_bb_ca, ref["bb_ca"]["v"][0]) < 5e-6
    assert rel(out.v_local_latents, ref["local_latents"]["v"][0]) < 5e-6
    assert rel(out.sequence_logits, ref["sequence"]["logits"][0]) < 5e-6
    assert rel(out.insertion_logits, ref["insertion"]["logits"][0]) < 1e-5


def test_decoder_parity(reference):
    import torch

    from jrfd4_proteina.features import DecoderInputs

    torch_model, model, torch_batch, batch = reference
    x, _ = _mid_trajectory_inputs(torch_model, torch_batch)
    with torch.no_grad():
        ref = torch_model.autoencoder.decode({**torch_batch, "z_latent": x["local_latents"], "ca_coors_nm": x["bb_ca"]})
    out = model.decoder(batch, DecoderInputs(x["bb_ca"][0].numpy(), x["local_latents"][0].numpy()))
    assert rel(out.coords_nm, ref["coors_nm"][0]) < 1e-4
    np.testing.assert_array_equal(np.asarray(out.residue_type), ref["residue_type"][0].numpy())
    np.testing.assert_array_equal(np.asarray(out.atom_mask), ref["atom_mask"][0].numpy())


def _replay(torch_model, torch_batch, batch, model, mode: str, nsteps: int, bands):
    """Run torch `full_simulation` recording every randn draw, then feed the draws to `integrate`."""
    import torch

    from jrfd4_proteina.sampling import Noise, _schedule, decode, integrate, release_sampling_config

    draws, randn = [], torch.randn

    def recording_randn(*args, **kwargs):
        out = randn(*args, **kwargs)
        draws.append(out.clone())
        return out

    depths = [d for width, d in bands for _ in range(width)]
    n = torch_batch["padding_mask"].shape[1]
    torch.manual_seed(0)
    torch.randn = recording_randn
    try:
        with torch.no_grad():
            x, extra = torch_model.fm.full_simulation(
                batch=dict(torch_batch), predict_for_sampling=_predict(torch_model), recycle_depths=depths,
                nsteps=nsteps, nsamples=1, n=n, sampling_model_args=_sampling_args(mode), device=torch.device("cpu"),
            )
    finally:
        torch.randn = randn

    config = release_sampling_config()
    config = dataclasses.replace(
        config, nsteps=nsteps, recycle_schedule=tuple(map(tuple, bands)),
        bb_ca=dataclasses.replace(config.bb_ca, sampling_mode=mode),
        local_latents=dataclasses.replace(config.local_latents, sampling_mode=mode),
    )
    stochastic = np.asarray(_schedule(config).stochastic)
    it = iter(draws)
    init_bb, group_offset, init_lat = next(it)[0].numpy(), next(it)[0].numpy(), next(it)[0].numpy()
    steps = [np.zeros((nsteps, n, 3), np.float32), np.zeros((nsteps, n, model.latent_dim), np.float32)]
    for s in range(nsteps):
        for m in (0, 1):
            if stochastic[s, m]:
                steps[m][s] = next(it)[0].numpy()
    assert next(it, None) is None, "the reference drew noise the port does not consume"
    offsets = np.zeros((n, 3), np.float32)
    offsets[0] = group_offset  # a monomer has one centering group
    noise = Noise(init_bb, init_lat, offsets, steps[0], steps[1], np.zeros((nsteps, batch.coord_group.size), np.float32))
    trajectory = integrate(model.flow, batch, noise, config, (model.noise_bb_ca, model.noise_local_latents))
    return x, extra, trajectory, decode(model.decoder, batch, trajectory, config)


def test_ode_sampling_parity_with_replayed_noise(reference):
    import torch
    from omegaconf import OmegaConf

    torch_model, model, torch_batch, batch = reference
    x, extra, trajectory, design = _replay(torch_model, torch_batch, batch, model, "vf", 24, [[4, 3], [10, 2], [10, 1]])
    assert rel(trajectory.x_bb_ca, x["bb_ca"][0]) < 1e-4
    assert rel(trajectory.x_local_latents, x["local_latents"][0]) < 5e-3

    torch_model.inf_cfg = OmegaConf.create({"args": {"condition_decoder_on_sequence_prediction": True}})
    with torch.no_grad():
        ref = torch_model.sample_formatting(
            x=x, ret_mode="coors37_n_aatype", batch={**torch_batch, "sequence_logits": extra["sequence_logits"]}
        )
    np.testing.assert_array_equal(np.asarray(design.residue_type), ref["residue_type"][0].numpy())
    np.testing.assert_array_equal(np.asarray(design.atom_mask), ref["atom_mask"][0].numpy())
    coords = np.asarray(design.coords) * np.asarray(design.atom_mask)[..., None]
    assert np.abs(coords - ref["coors"][0].numpy() * ref["atom_mask"][0].numpy()[..., None]).max() < 0.05


def test_tsr_sampling_consumes_the_reference_noise(reference):
    torch_model, model, torch_batch, batch = reference
    *_, design = _replay(torch_model, torch_batch, batch, model, "vf_tsr", 20, [[20, 1]])
    assert np.isfinite(np.asarray(design.coords)).all()


def test_save_load_round_trip_is_torch_free(reference, tmp_path):
    import jax

    from jrfd4_proteina import io

    _, model, _, _ = reference
    io.save(model, tmp_path / "model")
    code = (
        "import sys, jax; from jrfd4_proteina import io; "
        f"m = io.load({str(tmp_path / 'model')!r}); assert 'torch' not in sys.modules; "
        "print(sum(x.size for x in jax.tree.leaves(m) if hasattr(x, 'size')))"
    )
    n_params = int(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout)
    assert n_params == sum(x.size for x in jax.tree.leaves(model) if hasattr(x, "size"))
