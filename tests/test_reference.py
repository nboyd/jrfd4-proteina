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


def _with_mode(modality, mode: str):
    """Our modality config for the reference `sampling_mode`: `vf_tsr` is the release default, `vf` the plain ODE."""
    assert mode in ("vf", "vf_tsr")
    return modality if mode == "vf_tsr" else dataclasses.replace(modality, tsr=None, stochastic=None)


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

    from jrfd4_proteina.sampling import release_sampling_config, schedule

    fm = ProductSpaceFlowMatcher.__new__(ProductSpaceFlowMatcher)
    fm.data_modes = ["bb_ca", "local_latents"]
    for nsteps in (24, 60):
        ts, gt = ProductSpaceFlowMatcher.sample_schedule(fm, nsteps, _sampling_args("vf_tsr"))
        ours = schedule(dataclasses.replace(release_sampling_config(), nsteps=nsteps, recycle_schedule=((nsteps, 1),)))
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

    from jrfd4_proteina.sampling import Noise, StepNoise, decode, integrate, release_sampling_config, schedule

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
        bb_ca=_with_mode(config.bb_ca, mode),
        local_latents=_with_mode(config.local_latents, mode),
    )
    stochastic = np.asarray(schedule(config).stochastic)
    emit_bb = np.asarray(batch.padding_mask & ~batch.is_in_context_conditioning & ~batch.coord_cond_mask)
    n_groups = len(np.unique(np.asarray(batch.molecule_iid)[emit_bb]))  # one centering offset each, sorted
    it = iter(draws)
    init_bb = next(it)[0].numpy()
    group_offsets = [next(it)[0].numpy() for _ in range(n_groups)]
    init_lat = next(it)[0].numpy()
    steps = [np.zeros((nsteps, n, 3), np.float32), np.zeros((nsteps, n, model.latent_dim), np.float32)]
    for s in range(nsteps):
        for m in (0, 1):
            if stochastic[s, m]:
                steps[m][s] = next(it)[0].numpy()
    assert next(it, None) is None, "the reference drew noise the port does not consume"
    offsets = np.zeros((n, 3), np.float32)
    offsets[:n_groups] = group_offsets
    zeros = np.zeros((nsteps, n, 3), np.float32)
    priority = np.zeros((nsteps, batch.coord_group.size), np.float32)
    noise = Noise(init_bb, init_lat, offsets, StepNoise(steps[0], steps[1], priority, zeros, zeros))
    trajectory = integrate(model.flow, batch, noise, config, (model.noise_bb_ca, model.noise_local_latents))
    return x, extra, trajectory, decode(model.decoder, batch, trajectory, config)


def test_ode_sampling_parity_with_replayed_noise(reference):
    import torch
    from omegaconf import OmegaConf

    torch_model, model, torch_batch, batch = reference
    x, extra, trajectory, design = _replay(torch_model, torch_batch, batch, model, "vf", 24, [[4, 3], [10, 2], [10, 1]])
    assert rel(trajectory.final.bb_ca, x["bb_ca"][0]) < 1e-4
    assert rel(trajectory.final.local_latents, x["local_latents"][0]) < 5e-3

    torch_model.inf_cfg = OmegaConf.create({"args": {"condition_decoder_on_sequence_prediction": True}})
    with torch.no_grad():
        ref = torch_model.sample_formatting(
            x=x, ret_mode="coors37_n_aatype", batch={**torch_batch, "sequence_logits": extra["sequence_logits"]}
        )
    np.testing.assert_array_equal(np.asarray(design.residue_type), ref["residue_type"][0].numpy())
    np.testing.assert_array_equal(np.asarray(design.atom_mask), ref["atom_mask"][0].numpy())
    coords = np.asarray(design.coords) * np.asarray(design.atom_mask)[..., None]
    assert np.abs(coords - ref["coors"][0].numpy() * ref["atom_mask"][0].numpy()[..., None]).max() < 0.05


def test_binder_ode_sampling_parity_with_replayed_noise(pdl1_reference):
    import torch
    from omegaconf import OmegaConf

    torch_model, model, torch_batch, batch = pdl1_reference
    assert len(np.unique(np.asarray(batch.molecule_iid)[np.asarray(batch.padding_mask)])) == 2
    x, extra, trajectory, design = _replay(torch_model, torch_batch, batch, model, "vf", 24, [[4, 3], [10, 2], [10, 1]])
    assert rel(trajectory.final.bb_ca, x["bb_ca"][0]) < 1e-4
    assert rel(trajectory.final.local_latents, x["local_latents"][0]) < 5e-3

    torch_model.inf_cfg = OmegaConf.create({"args": {"condition_decoder_on_sequence_prediction": True}})
    with torch.no_grad():
        ref = torch_model.sample_formatting(
            x=x, ret_mode="coors37_n_aatype", batch={**torch_batch, "sequence_logits": extra["sequence_logits"]}
        )
    np.testing.assert_array_equal(np.asarray(design.residue_type), ref["residue_type"][0].numpy())
    mask = ref["atom_mask"][0].numpy()
    coords = np.asarray(design.coords) * mask[..., None]
    assert np.abs(coords - ref["coors"][0].numpy() * mask[..., None]).max() < 0.05


def test_featurize_spec_matches_reference_pipeline():
    import random
    from pathlib import Path

    import jax
    import torch
    import yaml
    from conftest import _reference_batch

    from jrfd4_proteina import convert

    def same(a, b):  # unset conditioning values are NaN
        return jax.tree.all(jax.tree.map(lambda x, y: np.array_equal(x, y, equal_nan=True), a, b))

    data = Path(__file__).parent / "data"
    spec = yaml.safe_load((data / "pdl1_binder.yaml").read_text())
    spec["pdl1_binder"]["input"] = str(data / spec["pdl1_binder"]["input"])
    (batch,) = convert.featurize(spec, seed=3)
    state = random.getstate()  # after the first call: importing the reference draws from `random`
    (again,) = convert.featurize(spec, seed=3)
    assert random.getstate() == state
    assert same(batch, again)
    assert len({int(convert.featurize(spec, seed=s)[0].padding_mask.sum()) for s in range(4)}) > 1

    random.seed(3)  # the reference pipeline draws from all three
    np.random.seed(3)
    torch.manual_seed(3)
    (ref,) = convert.batch_from_reference(_reference_batch(str(data / "pdl1_binder.yaml")))
    assert same(batch, ref)


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



def test_bfloat16_casts_only_the_flow_trunk(reference):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jrfd4_proteina.layers import Linear

    def dtypes(tree):  # of the floating-point parameters
        return {x.dtype for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))}

    f32, bf16 = np.dtype(jnp.float32), np.dtype(jnp.bfloat16)

    def is_linear(x):
        return isinstance(x, Linear)

    _, model, _, _ = reference
    mixed = model.bfloat16()
    assert dtypes(model) == {f32}  # the original is untouched
    trunk_linears = [x for x in jax.tree.leaves(mixed.flow.trunk, is_leaf=is_linear) if is_linear(x)]
    assert {x.weight.dtype for x in trunk_linears} == {bf16}
    for kept in (mixed.flow.ca_linear, mixed.flow.local_latents_linear, mixed.flow.sequence_readout,
                 mixed.flow.insertion_q_proj, mixed.flow.insertion_k_proj, mixed.decoder):
        assert dtypes(kept) == {f32}
    outside_linears = jax.tree.map(lambda x: None if is_linear(x) else x, mixed.flow, is_leaf=is_linear)
    assert dtypes(outside_linears) == {f32}  # norms, embeddings and other non-Linear parameters

def _flow_matcher():
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher

    fm = ProductSpaceFlowMatcher.__new__(ProductSpaceFlowMatcher)
    fm.data_modes = ["bb_ca", "local_latents"]
    fm.base_flow_matchers = {}
    return fm


def _ode_config(nsteps: int, bands):
    from jrfd4_proteina.sampling import release_sampling_config

    config = release_sampling_config()
    return dataclasses.replace(
        config, nsteps=nsteps, recycle_schedule=tuple(map(tuple, bands)),
        bb_ca=_with_mode(config.bb_ca, "vf"),
        local_latents=_with_mode(config.local_latents, "vf"),
    )


@pytest.mark.parametrize("partial", [{"partial_trajectory_fraction_steps": 0.4}, {"partial_t_backbone": 0.6},
                                     {"partial_t_latent": 0.3}, {"partial_t_backbone": 0.7, "partial_t_latent": 0.2}])
def test_partial_schedules_match_reference(partial):
    from rfproteina.search.vanilla import build_partial_schedule, resolve_partial_t

    from jrfd4_proteina.sampling import partial_schedule, release_sampling_config

    fm, args = _flow_matcher(), _sampling_args("vf_tsr")
    config = release_sampling_config()
    coupled = len(partial) == 1 and "partial_trajectory_fraction_steps" not in partial
    cfg_args = {"nsteps": config.nsteps, "couple_partial_t": coupled, **partial}
    pt = resolve_partial_t(cfg_args, fm, args)
    frac = partial.get("partial_trajectory_fraction_steps")
    nsteps = round(frac * config.nsteps) if frac else config.nsteps
    ts, gt = build_partial_schedule(pt, nsteps, fm, args, full_nsteps=config.nsteps if frac else None)

    sched, start = partial_schedule(config, fraction=frac, t_bb_ca=partial.get("partial_t_backbone"),
                                    t_local_latents=partial.get("partial_t_latent"))
    np.testing.assert_allclose(start, [pt["bb_ca"], pt["local_latents"]], rtol=0, atol=1e-6)
    for m, name in enumerate(fm.data_modes):
        np.testing.assert_allclose(np.append(sched.t[:, m], sched.t_next[-1, m]), ts[name].numpy(), rtol=0, atol=3e-7)
        assert rel(sched.g[:, m], gt[name].numpy()) < 1e-6


def test_partial_diffusion_parity(reference):
    import torch

    from jrfd4_proteina.sampling import State, StepNoise, partial_schedule, simulate

    torch_model, model, torch_batch, batch = reference
    bands = [[4, 3], [10, 2], [10, 1]]
    config = _ode_config(24, bands)
    sched, _ = partial_schedule(config, fraction=0.5)
    depths = [d for width, d in bands for _ in range(width)][-len(sched):]
    x, _ = _mid_trajectory_inputs(torch_model, torch_batch)
    ts = {name: torch.from_numpy(np.append(sched.t[:, m], sched.t_next[-1, m])) for m, name in
          enumerate(["bb_ca", "local_latents"])}
    gt = {name: torch.from_numpy(np.array(sched.g[:, m])) for m, name in enumerate(["bb_ca", "local_latents"])}
    args = _sampling_args("vf")
    with torch.no_grad():
        ref, ref_clean, _ = torch_model.fm.partial_simulation(
            batch=dict(torch_batch), x={k: v.clone() for k, v in x.items()}, predict_for_sampling=_predict(torch_model),
            recycle_depths=depths, nsteps=len(sched), ts=ts, gt=gt,
            simulation_step_params={k: v["simulation_step_params"] for k, v in args.items()},
            device=torch.device("cpu"),
        )

    n, steps = batch.padding_mask.shape[0], len(sched)
    zeros = np.zeros((steps, n, 3), np.float32)
    noise = StepNoise(zeros, np.zeros((steps, n, model.latent_dim), np.float32),
                      np.zeros((steps, batch.coord_group.size), np.float32), zeros, zeros)
    state = State(x["bb_ca"][0].numpy(), x["local_latents"][0].numpy())
    trajectory = simulate(model.flow, batch, state, sched, noise, config, model.noise_config)
    assert rel(trajectory.final.bb_ca, ref["bb_ca"][0]) < 1e-4
    assert rel(trajectory.final.local_latents, ref["local_latents"][0]) < 5e-3
    assert rel(trajectory.clean.bb_ca, ref_clean["bb_ca"][0]) < 1e-4


def test_inverse_folding_parity_with_replayed_noise(reference):
    import torch

    from jrfd4_proteina.sampling import State, StepNoise, schedule, simulate

    torch_model, model, torch_batch, batch = reference
    bands = [[4, 3], [10, 2], [10, 1]]
    config = _ode_config(24, bands)
    sched = schedule(config)
    depths = [d for width, d in bands for _ in range(width)]
    x, _ = _mid_trajectory_inputs(torch_model, torch_batch)
    target = torch_batch["coords_nm"][:, :, 1, :].float()
    ts = {name: torch.from_numpy(np.append(sched.t[:, m], sched.t_next[-1, m])) for m, name in
          enumerate(["bb_ca", "local_latents"])}
    gt = {name: torch.from_numpy(np.array(sched.g[:, m])) for m, name in enumerate(["bb_ca", "local_latents"])}
    args = _sampling_args("vf")

    draws, randn = [], torch.randn

    def recording_randn(*a, **kw):
        out = randn(*a, **kw)
        draws.append(out.clone())
        return out

    torch.manual_seed(0)
    torch.randn = recording_randn
    try:
        with torch.no_grad():
            ref, ref_clean, _ = torch_model.fm.partial_simulation(
                batch=dict(torch_batch), x={k: v.clone() for k, v in x.items()},
                predict_for_sampling=_predict(torch_model), recycle_depths=depths, nsteps=len(sched), ts=ts, gt=gt,
                simulation_step_params={k: v["simulation_step_params"] for k, v in args.items()},
                device=torch.device("cpu"), resample_target_backbone=True, target_backbone=target,
            )
    finally:
        torch.randn = randn

    n, steps = batch.padding_mask.shape[0], len(sched)
    assert len(draws) == 2 * steps, "one (noise, centering offset) pair per resample"
    marginal = np.stack([d[0].numpy() for d in draws[0::2]])
    offsets = np.zeros((steps, n, 3), np.float32)
    offsets[:, 0] = np.stack([d[0].numpy().reshape(3) for d in draws[1::2]])  # a monomer has one centering group
    zeros = np.zeros((steps, n, 3), np.float32)
    noise = StepNoise(zeros, np.zeros((steps, n, model.latent_dim), np.float32),
                      np.zeros((steps, batch.coord_group.size), np.float32), marginal, offsets)
    state = State(x["bb_ca"][0].numpy(), x["local_latents"][0].numpy())
    trajectory = simulate(model.flow, batch, state, sched, noise, config, model.noise_config,
                          fixed_bb_ca=target[0].numpy())
    assert rel(trajectory.final.bb_ca, ref["bb_ca"][0]) < 1e-6
    assert rel(trajectory.final.local_latents, ref["local_latents"][0]) < 5e-3
    assert rel(trajectory.clean.local_latents, ref_clean["local_latents"][0]) < 5e-3
