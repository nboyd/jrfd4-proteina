"""Reward-guided beam search over sampling trajectories (the reference's `BeamSearchStrategy`).

Start `beam_width` trajectories; at each checkpoint, branch every survivor `n_branch` times, integrate the
branches to the checkpoint step with fresh noise, score them, and keep the best `beam_width` states. A
branch is scored either by rolling it out to t=1 (fresh noise again) and decoding, or, with
`use_clean_prediction`, by decoding the network's clean prediction from its last step. Without a
checkpoint at the schedule end, the final survivors are rolled out once and returned unscored.

Rewards are maximized and run inside a jitted `lax.map` over candidates, so they must be JAX functions
`reward(design, *, key) -> reward | (reward, aux)`. Make them `eqx.Module`s that hold their models as
fields: they are passed to the jitted function as arguments, never closed over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jrfd4_proteina.batch import Batch
from jrfd4_proteina.model import DecoderTransformer, LocalLatentsTransformer
from jrfd4_proteina.sampling import (
    Design,
    NoiseConfig,
    RFD4Proteina,
    SamplingConfig,
    Schedule,
    State,
    Trajectory,
    decode,
    draw_noise,
    draw_steps,
    initial_state,
    schedule,
    simulate,
)


@dataclass(frozen=True)
class Checkpoint:
    """Score and prune at the end of step `step` (1-based, so `len(schedule)` is the schedule end)."""

    step: int
    reward: Callable[..., Any]  # (design, *, key) -> reward | (reward, aux); higher is better
    n_branch: int | None = None  # branches per survivor for the interval ending here (default: beam_search's)
    use_clean_prediction: bool = False  # score the last clean prediction instead of a rollout (not at the end)
    # Host-side check on each survivor's (reward, aux); the search stops when no survivor passes (not at the end)
    continue_if: Callable[[float, Any], bool] | None = None


@dataclass
class Scored:
    """All candidates evaluated at one checkpoint, stacked on a leading candidate axis."""

    step: int
    designs: Design
    rewards: Float[Array, " C"]  # non-finite rewards are recorded as -inf
    aux: Any
    selected: Int[np.ndarray, " W"]  # survivors, best first
    states: State  # [W] surviving integrator states at `step` (resume from here with `init=`, `start_step=`)


@dataclass
class BeamSearchResult:
    designs: Design | None  # [W] final beam, best first; None if `continue_if` stopped the search
    rewards: Float[Array, " W"] | None  # schedule-end rewards; None without a checkpoint at the schedule end
    checkpoints: list[Scored]


def _call_reward(reward, design: Design, key: PRNGKeyArray):
    out = reward(design, key=key)
    value, aux = out if isinstance(out, tuple) else (out, None)
    return jnp.asarray(value, jnp.float32), aux


@eqx.filter_jit
def _advance(
    model: LocalLatentsTransformer,
    batch: Batch,
    states: State,
    step: Schedule,
    keys: PRNGKeyArray,
    config: SamplingConfig,
    noise_config: tuple[NoiseConfig, NoiseConfig],
    chunk_size: int | None,
) -> Trajectory:
    """One schedule step for candidates `[C]`, each with fresh noise from its key."""

    def one(args):
        state, key = args
        steps = draw_steps(key, batch, nsteps=len(step), latent_dim=state.local_latents.shape[-1])
        return simulate(model, batch, state, step, steps, config, noise_config)

    return jax.lax.map(one, (states, keys), batch_size=chunk_size or keys.shape[0])


@eqx.filter_jit
def _decode_and_score(
    decoder: DecoderTransformer,
    batch: Batch,
    trajectories: Trajectory,
    reward,
    keys: PRNGKeyArray,
    config: SamplingConfig,
    chunk_size: int | None,
):
    def one(args):
        trajectory, key = args
        design = decode(decoder, batch, trajectory, config)
        if reward is None:
            return design, None
        return design, _call_reward(reward, design, key)

    return jax.lax.map(one, (trajectories, keys), batch_size=chunk_size or keys.shape[0])


def _run(model: RFD4Proteina, batch, states: State, sched: Schedule, start: int, end: int, key, config, chunk_size):
    """Integrate candidates `[C]` through steps `[start, end)`, stepwise: one compile per recycle depth."""
    n = states.bb_ca.shape[0]
    trajectory = None
    for s in range(start, end):
        keys = jax.random.split(jax.random.fold_in(key, s), n)
        trajectory = _advance(model.flow, batch, states, sched[s : s + 1], keys, config, model.noise_config,
                              chunk_size)
        states = trajectory.final
    return trajectory


def beam_search(
    model: RFD4Proteina,
    batch: Batch,
    config: SamplingConfig,
    checkpoints: Sequence[Checkpoint],
    *,
    key: PRNGKeyArray,
    beam_width: int = 4,
    n_branch: int = 4,
    sched: Schedule | None = None,
    init: State | None = None,
    start_step: int = 0,
    chunk_size: int | None = None,
) -> BeamSearchResult:
    """Beam search for one design problem.

    Args:
        sched: the schedule to search over (default `schedule(config)`; a `partial_schedule` for partial
            diffusion, with `init` the corrupted states).
        init: starting states, one `[N, ...]` (shared by the beam) or `[beam_width, N, ...]`; default fresh noise.
        start_step: the step `init` is at, to resume from `Scored.states`. Checkpoints at or before it are skipped.
        chunk_size: candidates per batch inside each jitted map (default: all at once).
    """
    sched = schedule(config) if sched is None else sched
    nsteps = len(sched)
    checkpoints = sorted(checkpoints, key=lambda c: c.step)
    steps = [c.step for c in checkpoints]
    if not checkpoints or len(set(steps)) != len(steps) or steps[0] < 1 or steps[-1] > nsteps:
        raise ValueError(f"need distinct checkpoint steps in [1, {nsteps}], got {steps}")
    end = checkpoints[-1]
    if end.step == nsteps and (end.use_clean_prediction or end.continue_if is not None):
        raise ValueError("a schedule-end checkpoint scores completed designs and cannot stop the search")
    if beam_width < 1 or n_branch < 1 or any(c.n_branch is not None and c.n_branch < 1 for c in checkpoints):
        raise ValueError("beam_width and n_branch must be positive")
    if not 0 <= start_step < nsteps:
        raise ValueError(f"start_step must be in [0, {nsteps})")

    init_key, loop_key, final_key = jax.random.split(key, 3)
    if init is None:
        if start_step:
            raise ValueError("resuming (start_step > 0) needs init")
        noise_keys = jax.random.split(init_key, beam_width)
        states = jax.vmap(
            lambda k: initial_state(batch, draw_noise(k, batch, nsteps=0, latent_dim=model.latent_dim),
                                    *model.noise_config)
        )(noise_keys)
    elif init.bb_ca.ndim == 2:
        states = jax.tree.map(lambda a: jnp.broadcast_to(a, (beam_width, *a.shape)), init)
    else:
        states = init
    if states.bb_ca.shape[0] != beam_width:
        raise ValueError(f"init has {states.bb_ca.shape[0]} states, expected beam_width={beam_width}")

    scored: list[Scored] = []
    designs = rewards = None
    start = start_step
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint.step <= start:
            continue
        ckey = jax.random.fold_in(loop_key, i)
        branches = checkpoint.n_branch or n_branch
        candidates = jax.tree.map(lambda a, b=branches: jnp.repeat(a, b, axis=0), states)
        trajectory = _run(model, batch, candidates, sched, start, checkpoint.step, jax.random.fold_in(ckey, 0),
                          config, chunk_size)
        if checkpoint.use_clean_prediction:
            evaluated = trajectory.at_clean_prediction()
        elif checkpoint.step < nsteps:
            evaluated = _run(model, batch, trajectory.final, sched, checkpoint.step, nsteps,
                             jax.random.fold_in(ckey, 1), config, chunk_size)
        else:
            evaluated = trajectory
        n = candidates.bb_ca.shape[0]
        score_keys = jax.random.split(jax.random.fold_in(ckey, 2), n)
        designs, (rewards, aux) = _decode_and_score(model.decoder, batch, evaluated, checkpoint.reward, score_keys,
                                                    config, chunk_size)
        rewards = jnp.where(jnp.isfinite(rewards), rewards, -jnp.inf)
        host_rewards = np.asarray(rewards)
        if np.isneginf(host_rewards).all():
            raise ValueError(f"every candidate at step {checkpoint.step} has a non-finite reward")
        selected = np.argsort(-host_rewards, kind="stable")[:beam_width]
        states = jax.tree.map(lambda a, s=selected: a[s], trajectory.final)
        scored.append(Scored(checkpoint.step, designs, rewards, aux, selected, states))
        start = checkpoint.step

        if checkpoint.continue_if is not None:
            host_aux = jax.device_get(aux)
            if not any(checkpoint.continue_if(float(host_rewards[j]), jax.tree.map(lambda a, j=j: a[j], host_aux))
                       for j in selected):
                return BeamSearchResult(None, None, scored)

    if start == nsteps:
        best = scored[-1].selected
        return BeamSearchResult(jax.tree.map(lambda a: a[best], designs), rewards[best], scored)
    trajectory = _run(model, batch, states, sched, start, nsteps, final_key, config, chunk_size)
    final, _ = _decode_and_score(model.decoder, batch, trajectory, None, jax.random.split(final_key, beam_width),
                                 config, chunk_size)
    return BeamSearchResult(final, None, scored)
