"""Torch-free tests of slicing, partial diffusion, inverse folding and beam search, on a toy flow network."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jrfd4_proteina import constants as C
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.model import DecoderOutput, FlowOutput
from jrfd4_proteina.sampling import (
    TSR,
    NoiseConfig,
    RFD4Proteina,
    State,
    corrupt,
    decode,
    draw_noise,
    initial_state,
    inverse_fold,
    partial_schedule,
    release_sampling_config,
    sample,
    schedule,
    simulate,
)
from jrfd4_proteina.search import Checkpoint, beam_search

N, L = 12, 4


def toy_batch() -> Batch:
    shapes = {
        "sequence_probabilities": (N, len(C.SEQUENCE_PROBABILITY_RESIDUES)), "cath": (N, 3), "chain_centroid": (N, 3),
        "chirality": (N, 3 * C.MAX_CHIRAL_CENTERS), "chiral_frame_atoms": (N, C.MAX_CHIRAL_CENTERS, 4),
        "chiral_slot_owner": (N, C.MAX_CHIRAL_CENTERS), "cond_coords": (N, 37, 3), "cond_internal_coords": (N, 37, 3),
    }
    pair = {"bond_mask", "bond_order", "is_aromatic", "token_bond_path_length", "canonical_polymer_ordering",
            "na_ss_pair", "na_ss_pair_mask"}
    atom = {"atom37_mask", "charge", "element", "coord_group", "rasa", "hbond", "pistack", "pication", "saltbridge",
            "hydrophobic", "metal_coord", "hotspot"}
    fields = {}
    for name in Batch.__dataclass_fields__:
        shape = shapes.get(name, (N, N) if name in pair else (N, 37) if name in atom else (N,))
        fields[name] = np.zeros(shape, np.int32)
    fields["padding_mask"] = np.arange(N) < N - 2  # two padding tokens
    fields["supports_local_latent"] = np.ones(N, bool)
    fields["is_in_context_conditioning"] = np.zeros(N, bool)
    fields["coord_cond_mask"] = np.zeros(N, bool)
    fields["coord_group"] = np.full((N, 37), C.UNKNOWN_COORDS, np.int32)
    return Batch(**fields)


class ToyFlow(eqx.Module):
    """Velocities toward a fixed target, perturbed by the recycle depth."""

    target: jax.Array

    def __call__(self, batch, inputs, n_recycle):
        v_bb = (self.target - inputs.x_bb_ca) / (1.0 - inputs.t_bb_ca + 0.05) + 0.01 * n_recycle
        v_lat = jnp.tanh(inputs.x_local_latents) - inputs.x_local_latents
        logits = jnp.concatenate([inputs.x_local_latents, jnp.zeros((N, C.N_RESIDUE_TYPES - L))], -1)
        return FlowOutput(v_bb, v_lat, logits, jnp.zeros((N, N)))


class ToyDecoder(eqx.Module):
    allowed_residue_types: jax.Array

    def __call__(self, batch, inputs, required_atom_mask=None):
        emit = batch.latent_mask
        coords = jnp.broadcast_to(inputs.ca_coors_nm[:, None], (N, 37, 3)) * emit[:, None, None]
        atom_mask = (jnp.arange(37) == 1)[None] & emit[:, None]
        return DecoderOutput(coords, (batch.cond_sequence * emit).astype(jnp.int32), atom_mask, emit,
                             jnp.zeros((N, C.N_RESIDUE_TYPES)))


def toy_model() -> RFD4Proteina:
    target = jax.random.normal(jax.random.key(1), (N, 3))
    return RFD4Proteina(ToyFlow(target), ToyDecoder(jnp.ones((C.N_TOKEN_TYPES, C.N_RESIDUE_TYPES), bool)),
                        NoiseConfig(zero_com=True, centering_scale=0.2), NoiseConfig(False, None), L)


def toy_config(nsteps=24):
    config = release_sampling_config()
    return dataclasses.replace(config, nsteps=nsteps, recycle_schedule=((4, 3), (10, 2), (10, 1)))


class Contraction(eqx.Module):
    """Reward: negative mean CA distance from `center` (an array field, so it is traced, not baked in)."""

    center: jax.Array

    def __call__(self, design, *, key):
        ca = design.coords[:, 1]
        r = jnp.linalg.norm(ca - self.center, axis=-1)
        return -(r * design.atom_mask[:, 1]).sum() / design.atom_mask[:, 1].sum(), {"radius": r.max()}


# --------------------------------------------------------------------------- schedules


def test_schedule_slices_keep_their_recycle_bands():
    s = schedule(toy_config())
    assert s.bands == ((4, 3), (10, 2), (10, 1))
    assert s[2:16].bands == ((2, 3), (10, 2), (2, 1))
    assert len(s[20:]) == 4 and s[20:21].depths == (1,)


def test_fractional_partial_schedule_is_the_tail_of_the_full_one():
    config = toy_config()
    full = schedule(config)
    sched, start = partial_schedule(config, fraction=0.25)
    assert len(sched) == 6 and sched.depths == full.depths[18:]
    np.testing.assert_array_equal(sched.t, full.t[18:])
    np.testing.assert_array_equal(start, full.t[18])


def test_partial_schedule_from_start_times():
    config = toy_config()
    sched, start = partial_schedule(config, t_bb_ca=0.6, t_local_latents=0.3)
    np.testing.assert_array_equal(start, np.float32([0.6, 0.3]))
    assert len(sched) == config.nsteps and sched.depths == schedule(config).depths
    assert np.all(sched.t[0] >= start)  # the reference's expanded-schedule estimate starts at or after them
    np.testing.assert_array_equal(sched.t_next[-1], [1.0, 1.0])
    assert np.all(np.diff(sched.t, axis=0) > 0)

    _, coupled = partial_schedule(config, t_bb_ca=0.6)  # latent time read off the same schedule position
    assert coupled[0] == pytest.approx(0.6, abs=1e-3) and 0 < coupled[1] < 1
    with pytest.raises(ValueError):
        partial_schedule(config, t_bb_ca=1.0, t_local_latents=0.5)
    with pytest.raises(ValueError):
        partial_schedule(config, fraction=0.5, t_bb_ca=0.5)


# --------------------------------------------------------------------------- integration


def test_simulating_in_pieces_matches_one_run():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    noise = draw_noise(jax.random.key(0), batch, nsteps=config.nsteps, latent_dim=L)
    sched = schedule(config)
    state = initial_state(batch, noise, *model.noise_config)
    whole = simulate(model.flow, batch, state, sched, noise.steps, config, model.noise_config)
    piece = state
    for a, b in [(0, 3), (3, 17), (17, 24)]:
        steps = jax.tree.map(lambda x, a=a, b=b: x[a:b], noise.steps)
        trajectory = simulate(model.flow, batch, piece, sched[a:b], steps, config, model.noise_config)
        piece = trajectory.final
    np.testing.assert_allclose(piece.bb_ca, whole.final.bb_ca, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(piece.local_latents, whole.final.local_latents, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(trajectory.clean.bb_ca, whole.clean.bb_ca, rtol=1e-5, atol=1e-6)


def test_temperatures_can_be_traced_and_vmapped():
    model, batch, config, key = toy_model(), toy_batch(), toy_config(), jax.random.key(0)

    def at(k, sigma, scale):
        bb_ca, latents = config.bb_ca, config.local_latents
        return dataclasses.replace(
            config,
            bb_ca=dataclasses.replace(bb_ca, tsr=TSR(k, bb_ca.tsr.sigma),
                                      stochastic=dataclasses.replace(bb_ca.stochastic, noise_scale=scale)),
            local_latents=dataclasses.replace(latents, tsr=TSR(latents.tsr.k, sigma)),
        )

    settings = [(4.0, 12.0, 1.0), (16.0, 3.0, 0.5)]
    swept = jax.vmap(lambda k, sigma, scale: sample(model, batch, at(k, sigma, scale), key=key).trajectory.final)(
        *map(jnp.array, zip(*settings)))
    for i, setting in enumerate(settings):
        one = sample(model, batch, at(*setting), key=key).trajectory.final  # Python floats: compiled in
        np.testing.assert_allclose(swept.bb_ca[i], one.bb_ca, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(swept.local_latents[i], one.local_latents, rtol=1e-5, atol=1e-6)
    assert not np.allclose(swept.bb_ca[0], swept.bb_ca[1])  # the settings matter


def test_corrupt_interpolates_from_the_initial_noise():
    model, batch = toy_model(), toy_batch()
    noise = draw_noise(jax.random.key(0), batch, nsteps=0, latent_dim=L)
    clean = State(jnp.ones((N, 3)), jnp.ones((N, L)))
    x0 = initial_state(batch, noise, *model.noise_config)
    x = corrupt(batch, clean, jnp.array([0.0, 1.0]), noise, model.noise_config)
    np.testing.assert_allclose(x.bb_ca, x0.bb_ca)
    np.testing.assert_allclose(x.local_latents, np.asarray(batch.padding_mask)[:, None] * np.ones((N, L)))


def test_inverse_folding_keeps_the_backbone():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    ca = 10.0 * jax.random.normal(jax.random.key(3), (N, 3))
    design = inverse_fold(model, batch, config, ca, key=jax.random.key(0))
    emit = np.asarray(batch.padding_mask)[:, None]
    np.testing.assert_allclose(design.coords[:, 1], np.asarray(ca) * emit, rtol=1e-6, atol=1e-5)
    assert np.isfinite(np.asarray(design.trajectory.final.local_latents)).all()


# --------------------------------------------------------------------------- beam search


def test_beam_search_keeps_the_best_rollouts():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    reward = Contraction(jnp.zeros(3))
    result = beam_search(model, batch, config, [Checkpoint(8, reward), Checkpoint(24, reward, n_branch=3)],
                         key=jax.random.key(0), beam_width=2, n_branch=2)
    first, last = result.checkpoints
    assert first.rewards.shape == (4,) and last.rewards.shape == (6,)
    assert first.states.bb_ca.shape == (2, N, 3)
    np.testing.assert_array_equal(result.rewards, np.sort(np.asarray(last.rewards))[::-1][:2])
    assert result.designs.coords.shape == (2, N, 37, 3)
    # Survivors' designs are the scored schedule-end designs.
    np.testing.assert_array_equal(result.designs.coords, np.asarray(last.designs.coords)[last.selected])
    assert last.aux["radius"].shape == (6,)

    again = beam_search(model, batch, config, [Checkpoint(8, reward), Checkpoint(24, reward, n_branch=3)],
                        key=jax.random.key(0), beam_width=2, n_branch=2)
    np.testing.assert_array_equal(again.rewards, result.rewards)


def test_beam_search_without_an_end_checkpoint_finishes_survivors_unscored():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    reward = Contraction(jnp.zeros(3))
    result = beam_search(model, batch, config, [Checkpoint(6, reward, use_clean_prediction=True)],
                         key=jax.random.key(0), beam_width=3, n_branch=2)
    assert result.rewards is None and result.designs.coords.shape == (3, N, 37, 3)
    (scored,) = result.checkpoints
    # Clean-prediction scoring decodes the last network prediction, not a rollout.
    np.testing.assert_array_equal(scored.designs.trajectory.final.bb_ca, scored.designs.trajectory.clean.bb_ca)


def test_beam_search_stops_when_no_survivor_passes():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    reward = Contraction(jnp.zeros(3))
    stop = Checkpoint(6, reward, use_clean_prediction=True, continue_if=lambda r, aux: aux["radius"] < 0)
    result = beam_search(model, batch, config, [stop, Checkpoint(24, reward)], key=jax.random.key(0), beam_width=2)
    assert result.designs is None and len(result.checkpoints) == 1


def test_beam_search_resumes_from_saved_states_and_partial_schedules():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    reward = Contraction(jnp.zeros(3))
    first = beam_search(model, batch, config, [Checkpoint(10, reward)], key=jax.random.key(0), beam_width=2)
    saved = first.checkpoints[0].states
    resumed = beam_search(model, batch, config, [Checkpoint(10, reward), Checkpoint(24, reward)],
                          key=jax.random.key(1), beam_width=2, init=saved, start_step=10)
    assert [c.step for c in resumed.checkpoints] == [24]

    sched, start = partial_schedule(config, fraction=0.5)
    clean = State(first.designs.trajectory.final.bb_ca[0], first.designs.trajectory.final.local_latents[0])
    noise = draw_noise(jax.random.key(2), batch, nsteps=0, latent_dim=L)
    init = corrupt(batch, clean, start, noise, model.noise_config)
    refined = beam_search(model, batch, config, [Checkpoint(len(sched), reward)], key=jax.random.key(3),
                          beam_width=2, sched=sched, init=init)
    assert refined.rewards.shape == (2,)


def test_beam_search_rejects_bad_checkpoints():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    reward = Contraction(jnp.zeros(3))
    for checkpoints in ([], [Checkpoint(0, reward)], [Checkpoint(25, reward)],
                        [Checkpoint(24, reward, use_clean_prediction=True)]):
        with pytest.raises(ValueError):
            beam_search(model, batch, config, checkpoints, key=jax.random.key(0))


def test_decode_of_a_clean_prediction_uses_its_coordinates():
    model, batch, config = toy_model(), toy_batch(), toy_config()
    noise = draw_noise(jax.random.key(0), batch, nsteps=config.nsteps, latent_dim=L)
    sched = schedule(config)[:5]
    state = initial_state(batch, noise, *model.noise_config)
    steps = jax.tree.map(lambda x: x[:5], noise.steps)
    trajectory = simulate(model.flow, batch, state, sched, steps, config, model.noise_config)
    design = decode(model.decoder, batch, trajectory.at_clean_prediction(), config)
    np.testing.assert_allclose(design.coords[:, 1], np.asarray(trajectory.clean.bb_ca) * 10.0, rtol=1e-6)
