"""Flow-matching sampler (the reference's `ProductSpaceFlowMatcher.full_simulation` / `partial_simulation`) and
decoding.

Two modalities are integrated jointly from t=0 (noise) to t=1 (data): CA coordinates (nm) and local
latents. Each step evaluates the flow network at a recycle depth fixed by a banded schedule; each band
is one `lax.scan`, so the whole sampler is jittable and reverse-mode differentiable. `simulate` runs any
contiguous slice of a `Schedule` from a given `State`, which is what partial diffusion and search build on.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jrfd4_proteina import constants as C
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.features import DecoderInputs, FlowInputs
from jrfd4_proteina.layers import matmul_dtype
from jrfd4_proteina.model import DecoderTransformer, FlowOutput, LocalLatentsTransformer


@dataclass(frozen=True)
class ModalityConfig:
    """Time schedule, noise schedule and step rule for one modality (static)."""

    schedule: str  # "uniform" | "power" | "log"
    schedule_p: float
    tail_steps: int  # final intervals sampled at `tail_step_multiplier` x density (0: off)
    tail_step_multiplier: int
    gt: str  # "1/t" | "tan" | "1-t/t" | "ode"
    sampling_mode: str  # "vf_tsr" (temporal score rescaling) | "vf" (plain ODE)
    sc_scale_noise: float
    t_lim_ode: float  # deterministic above this time
    t_lim_ode_below: float  # deterministic below this time
    tsr_k: float
    tsr_sigma: float


@dataclass(frozen=True)
class NoiseConfig:
    """Initial-noise options for one modality, fixed by training (read from the checkpoint)."""

    zero_com: bool
    centering_scale: float | None  # stochastic x_0 offset per `molecule_iid` group (None: off)


@dataclass(frozen=True)
class SamplingConfig:
    bb_ca: ModalityConfig
    local_latents: ModalityConfig
    nsteps: int
    recycle_schedule: tuple[tuple[int, int], ...]  # (number of steps, trunk passes per step) bands
    decode_with_predicted_sequence: bool


def release_sampling_config() -> SamplingConfig:
    """The reference defaults: `generation/inference_base` with the medium front-loaded recycle schedule."""
    common = {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0, "t_lim_ode": 0.95, "t_lim_ode_below": 0.02,
                  "tail_steps": 10, "tail_step_multiplier": 2}
    return SamplingConfig(
        bb_ca=ModalityConfig(schedule="log", schedule_p=2.0, gt="1/t", tsr_k=16.0, tsr_sigma=3.0, **common),
        local_latents=ModalityConfig(schedule="power", schedule_p=2.0, gt="tan", tsr_k=4.0, tsr_sigma=12.0, **common),
        nsteps=60,
        recycle_schedule=((5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1)),
        decode_with_predicted_sequence=True,
    )


# --------------------------------------------------------------------------- schedules (host, float32)


def _base_schedule(mode: str, nsteps: int, p: float, tsr_safe: bool) -> np.ndarray:
    eps = np.float32(1e-3)
    f32 = np.float32
    if mode == "uniform":
        return np.linspace(eps if tsr_safe else 0, 1, nsteps + 1, dtype=f32)
    if mode == "power":
        start = eps ** (1 / p) if tsr_safe else 0.0
        return np.linspace(start, 1, nsteps + 1, dtype=f32) ** f32(p)
    if mode == "log":
        t = (1.0 - (f32(10.0) ** np.linspace(-p, eps if tsr_safe else 0.0, nsteps + 1, dtype=f32))[::-1]).astype(f32)
        t = t - ((t.min() - eps) if tsr_safe else t.min())
        return (t / t.max()).astype(f32)
    raise ValueError(f"unknown schedule {mode!r}")


def time_schedule(cfg: ModalityConfig, nsteps: int) -> np.ndarray:
    """`nsteps + 1` times from ~0 to 1 (starting at 1e-3 under TSR, where t=0 is singular)."""
    tsr_safe = cfg.sampling_mode == "vf_tsr"
    if not cfg.tail_steps:
        return _base_schedule(cfg.schedule, nsteps, cfg.schedule_p, tsr_safe)
    m = cfg.tail_step_multiplier
    base_steps = nsteps - (m - 1) * cfg.tail_steps
    if m < 2 or base_steps < cfg.tail_steps:
        raise ValueError("need tail_step_multiplier >= 2 and nsteps >= tail_steps * tail_step_multiplier")
    tail_start = (base_steps - cfg.tail_steps) * m
    indices = [*range(0, tail_start, m), *range(tail_start, base_steps * m + 1)]
    return _base_schedule(cfg.schedule, base_steps * m, cfg.schedule_p, tsr_safe)[indices]


def tsr_ratio(t, k: float, sigma: float):
    snr = t**2 / (1 - t) ** 2
    return (snr * sigma**2 + 1) / (snr * sigma**2 / k + 1)


def noise_schedule(cfg: ModalityConfig, t: np.ndarray) -> np.ndarray:
    """g(t) of the SDE `dx = [v + g s] dt + sqrt(2 g) dW`, at the step start times `t`."""
    if cfg.gt == "ode":
        return np.zeros_like(t)
    tc = np.clip(t, 0, np.float32(1 - 1e-5))
    num = {"1-t/t": 1.0 - tc, "1/t": np.ones_like(tc), "tan": np.sin((1.0 - tc) * np.pi / 2.0) * (np.pi / 2.0)}
    den = {"1-t/t": tc, "1/t": tc, "tan": np.cos((1.0 - tc) * np.pi / 2.0)}
    g = (num[cfg.gt] / (den[cfg.gt] + np.float32(1e-2))).astype(np.float32)
    if cfg.sampling_mode == "vf_tsr":
        g = g / tsr_ratio(t, cfg.tsr_k, cfg.tsr_sigma) ** 2  # ratio uses the unclamped times
    return g.astype(np.float32)


class Schedule(eqx.Module):
    """Per-step times, noise scales and recycle depths, columns (bb_ca, local_latents). Slice with `[a:b]`."""

    t: Float[Array, "S 2"]  # step start times
    t_next: Float[Array, "S 2"]
    g: Float[Array, "S 2"]
    stochastic: Bool[Array, "S 2"]
    depths: tuple[int, ...] = eqx.field(static=True)  # trunk passes per step

    def __getitem__(self, steps: slice) -> Schedule:
        return Schedule(self.t[steps], self.t_next[steps], self.g[steps], self.stochastic[steps], self.depths[steps])

    def __len__(self) -> int:
        return len(self.depths)

    @property
    def bands(self) -> tuple[tuple[int, int], ...]:
        """Runs of equal depth, as (number of steps, depth)."""
        return tuple((len(list(run)), depth) for depth, run in itertools.groupby(self.depths))


def _schedule_from_times(config: SamplingConfig, times: tuple[np.ndarray, np.ndarray], depths) -> Schedule:
    gs, stoch = [], []
    for cfg, t in zip((config.bb_ca, config.local_latents), times, strict=True):
        gs.append(noise_schedule(cfg, t[:-1]))
        stoch.append((t[:-1] <= cfg.t_lim_ode) & (t[:-1] >= cfg.t_lim_ode_below) & (cfg.sampling_mode == "vf_tsr"))
    t = np.stack(times, -1)
    return Schedule(t[:-1], t[1:], np.stack(gs, -1), np.stack(stoch, -1), tuple(depths))


def _recycle_depths(config: SamplingConfig) -> tuple[int, ...]:
    depths = tuple(depth for width, depth in config.recycle_schedule for _ in range(width))
    if len(depths) != config.nsteps:
        raise ValueError(f"recycle schedule covers {len(depths)} steps, expected {config.nsteps}")
    return depths


def schedule(config: SamplingConfig) -> Schedule:
    """The full `config.nsteps`-step schedule from ~0 to 1."""
    times = (time_schedule(config.bb_ca, config.nsteps), time_schedule(config.local_latents, config.nsteps))
    return _schedule_from_times(config, times, _recycle_depths(config))


def partial_schedule(
    config: SamplingConfig,
    *,
    fraction: float | None = None,
    t_bb_ca: float | None = None,
    t_local_latents: float | None = None,
) -> tuple[Schedule, np.ndarray]:
    """Schedule and start times `[2]` for partial diffusion (the reference's `resolve_partial_t` and
    `build_partial_schedule`).

    Either `fraction`, to run the final `round(fraction * nsteps)` steps of the full schedule
    (`partial_trajectory_fraction_steps`), or start times, to run `nsteps` steps from them. Given only one
    time, the other is read at the same position of a dense schedule (`couple_partial_t`).
    """
    if fraction is not None:
        if t_bb_ca is not None or t_local_latents is not None:
            raise ValueError("fraction is mutually exclusive with start times")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        remaining = round(fraction * config.nsteps)
        if remaining < 1:
            raise ValueError(f"fraction={fraction} with nsteps={config.nsteps} rounds to zero steps")
        sched = schedule(config)[config.nsteps - remaining :]
        return sched, _check_start(np.asarray(sched.t[0]))

    if t_bb_ca is None and t_local_latents is None:
        raise ValueError("give fraction or at least one start time")
    if t_bb_ca is None or t_local_latents is None:
        ref_nsteps = 10000
        refs = [time_schedule(cfg, ref_nsteps) for cfg in (config.bb_ca, config.local_latents)]
        which, value = (0, t_bb_ca) if t_bb_ca is not None else (1, t_local_latents)
        above = refs[which] >= value - 1e-6
        k = int(np.argmax(above)) if above.any() else ref_nsteps
        start = np.array([refs[0][k], refs[1][k]], np.float32)
    else:
        start = np.array([t_bb_ca, t_local_latents], np.float32)
    _check_start(start)

    n = config.nsteps
    dense_n = max(n * 100, 1000)
    times = []
    for cfg, t0 in zip((config.bb_ca, config.local_latents), start, strict=True):
        above = time_schedule(cfg, dense_n) >= t0 - 1e-6
        n_above = max(dense_n - (int(np.argmax(above)) if above.any() else dense_n), 1)
        times.append(time_schedule(cfg, round(n * dense_n / n_above))[-(n + 1) :])
    return _schedule_from_times(config, tuple(times), _recycle_depths(config)), start


def _check_start(start: np.ndarray) -> np.ndarray:
    if (start >= 1.0).any():
        raise ValueError(f"partial-diffusion start times must be < 1, got {start}")
    return start


# --------------------------------------------------------------------------- noise


class StepNoise(eqx.Module):
    """Per-step random draws (standard normal / uniform)."""

    bb_ca: Float[Array, "S N 3"]
    local_latents: Float[Array, "S N L"]
    coord_group_priority: Float[Array, "S K"]  # per-call coordinate-group relabelling
    # Forward-marginal backbone draws, used only when inverse folding (`fixed_bb_ca`)
    marginal_bb_ca: Float[Array, "S N 3"]
    marginal_offsets: Float[Array, "S G 3"]


class Noise(eqx.Module):
    """Every random draw of one sampling run."""

    bb_ca: Float[Array, "N 3"]
    local_latents: Float[Array, "N L"]
    group_offsets: Float[Array, "G 3"]  # one per centering group, in sorted group order
    steps: StepNoise


def draw_steps(key: PRNGKeyArray, batch: Batch, *, nsteps: int, latent_dim: int) -> StepNoise:
    n = batch.padding_mask.shape[0]
    keys = jax.random.split(key, 5)
    return StepNoise(
        bb_ca=jax.random.normal(keys[0], (nsteps, n, 3)),
        local_latents=jax.random.normal(keys[1], (nsteps, n, latent_dim)),
        coord_group_priority=jax.random.uniform(keys[2], (nsteps, batch.coord_group.size)),
        marginal_bb_ca=jax.random.normal(keys[3], (nsteps, n, 3)),
        marginal_offsets=jax.random.normal(keys[4], (nsteps, n, 3)),
    )


def draw_noise(key: PRNGKeyArray, batch: Batch, *, nsteps: int, latent_dim: int) -> Noise:
    n = batch.padding_mask.shape[0]
    keys = jax.random.split(key, 4)
    return Noise(
        bb_ca=jax.random.normal(keys[0], (n, 3)),
        local_latents=jax.random.normal(keys[1], (n, latent_dim)),
        group_offsets=jax.random.normal(keys[2], (n, 3)),
        steps=draw_steps(keys[3], batch, nsteps=nsteps, latent_dim=latent_dim),
    )


def masked_mean(x: Float[Array, "N D"], mask: Bool[Array, " N"]) -> Float[Array, " D"]:
    return (x * mask[:, None]).sum(0) / jnp.maximum(mask.sum(), 1)


class State(eqx.Module):
    """The integrated state between steps (sampler units: CA in nm)."""

    bb_ca: Float[Array, "N 3"]
    local_latents: Float[Array, "N L"]


def _bb_ca_noise(batch: Batch, eps: Float[Array, "N 3"], offsets: Float[Array, "G 3"], cfg: NoiseConfig):
    """x_0 for the CA modality: masked, optionally zero-COM, plus a stochastic offset per centering group."""
    emit_bb = batch.design_mask
    x_bb = eps * emit_bb[:, None]
    if cfg.zero_com:
        x_bb = (x_bb - masked_mean(x_bb, emit_bb)) * emit_bb[:, None]
    if cfg.centering_scale is not None:
        offset_mask = emit_bb & ~batch.coord_cond_mask
        ids = batch.molecule_iid.astype(jnp.int32)
        big = jnp.iinfo(jnp.int32).max
        groups = jnp.unique(jnp.where(offset_mask, ids, big), size=ids.shape[0], fill_value=big)
        offset = offsets[jnp.searchsorted(groups, ids)] * offset_mask[:, None]
        x_bb = (x_bb + cfg.centering_scale * offset) * emit_bb[:, None]
    return x_bb


def initial_state(batch: Batch, noise: Noise, bb_ca: NoiseConfig, local_latents: NoiseConfig) -> State:
    emit_lat = batch.latent_mask
    if local_latents.zero_com or local_latents.centering_scale is not None:
        raise NotImplementedError("latent centering is not used by the release")
    return State(_bb_ca_noise(batch, noise.bb_ca, noise.group_offsets, bb_ca), noise.local_latents * emit_lat[:, None])


def corrupt(
    batch: Batch, clean: State, t: Float[Array, " 2"], noise: Noise, noise_config: tuple[NoiseConfig, NoiseConfig]
) -> State:
    """The forward process `(1 - t) x_0 + t x_1` at per-modality times `t`, with `x_0 = initial_state(...)`."""
    x0 = initial_state(batch, noise, *noise_config)
    emit_bb, emit_lat = batch.design_mask, batch.latent_mask
    return State(
        (1.0 - t[0]) * x0.bb_ca + t[0] * clean.bb_ca * emit_bb[:, None],
        (1.0 - t[1]) * x0.local_latents + t[1] * clean.local_latents * emit_lat[:, None],
    )


# --------------------------------------------------------------------------- integration


def _step(cfg: ModalityConfig, x, v, t, t_next, g, stochastic, eps, emit):
    dt = t_next - t
    if cfg.sampling_mode == "vf":
        dx = v * dt
    else:
        scaled_score = (t * v - x) / (1.0 - t) * tsr_ratio(t, cfg.tsr_k, cfg.tsr_sigma)
        v_scaled = (x + (1.0 - t) * scaled_score) / t
        drift = jnp.where(stochastic, v_scaled + g * scaled_score, v_scaled)
        diffusion = jnp.where(stochastic, jnp.sqrt(2 * g * cfg.sc_scale_noise * dt), 0.0)
        dx = drift * dt + diffusion * eps
    return (x + dx) * emit[:, None]


class Trajectory(eqx.Module):
    final: State
    clean: State  # the network's clean prediction (x_1) at the last step
    last: FlowOutput  # network output at the last step

    def at_clean_prediction(self) -> Trajectory:
        """This trajectory with its final state replaced by the last clean prediction (for decoding it)."""
        return Trajectory(self.clean, self.clean, self.last)


@eqx.filter_jit
def simulate(
    model: LocalLatentsTransformer,
    batch: Batch,
    state: State,
    schedule: Schedule,
    steps: StepNoise,
    config: SamplingConfig,
    noise_config: tuple[NoiseConfig, NoiseConfig],
    fixed_bb_ca: Float[Array, "N 3"] | None = None,
) -> Trajectory:
    """Integrate `state` through every step of `schedule` with the given draws (deterministic).

    With `fixed_bb_ca` (nm) only the latents are generated (the reference's `resampled_forward_marginal`
    inverse folding): before each step the CA state is redrawn from its forward marginal around
    `fixed_bb_ca`, and it is set to `fixed_bb_ca` at t=1.
    """
    if len(schedule) == 0:
        raise ValueError("empty schedule")
    emit_bb, emit_lat = batch.design_mask, batch.latent_mask
    x = (state.bb_ca, state.local_latents)
    last = clean = None
    start = 0
    for width, depth in schedule.bands:
        band = slice(start, start + width)
        xs = (
            (schedule.t[band], schedule.t_next[band], schedule.g[band], schedule.stochastic[band]),
            jax.tree.map(lambda a, band=band: a[band], steps),
        )

        def body(x, step, depth=depth):
            (t, t_next, g, stochastic), eps = step
            x_bb, x_lat = x
            if fixed_bb_ca is not None:
                x0 = _bb_ca_noise(batch, eps.marginal_bb_ca, eps.marginal_offsets, noise_config[0])
                x_bb = (1.0 - t[0]) * x0 + t[0] * fixed_bb_ca * emit_bb[:, None]
            out = model(batch, FlowInputs(x_bb, x_lat, t[0], t[1], eps.coord_group_priority), depth)
            clean = (
                (x_bb + (1.0 - t[0]) * out.v_bb_ca) * emit_bb[:, None],
                (x_lat + (1.0 - t[1]) * out.v_local_latents) * emit_lat[:, None],
            )
            x_bb = _step(config.bb_ca, x_bb, out.v_bb_ca, t[0], t_next[0], g[0], stochastic[0], eps.bb_ca, emit_bb)
            x_lat = _step(config.local_latents, x_lat, out.v_local_latents, t[1], t_next[1], g[1], stochastic[1],
                          eps.local_latents, emit_lat)
            if fixed_bb_ca is not None:
                x_bb = jnp.where(t_next[0] >= 1.0, fixed_bb_ca * emit_bb[:, None], x_bb)
            return (x_bb, x_lat), (out, clean)

        x, (outs, cleans) = jax.lax.scan(body, x, xs)
        last, clean = jax.tree.map(lambda a: a[-1], (outs, cleans))
        start += width
    return Trajectory(State(*x), State(*clean), last)


@eqx.filter_jit
def integrate(
    model: LocalLatentsTransformer,
    batch: Batch,
    noise: Noise,
    config: SamplingConfig,
    noise_config: tuple[NoiseConfig, NoiseConfig],
) -> Trajectory:
    """Run the full sampler from given draws (deterministic)."""
    state = initial_state(batch, noise, *noise_config)
    return simulate(model, batch, state, schedule(config), noise.steps, config, noise_config)


class Design(eqx.Module):
    coords: Float[Array, "N 37 3"]  # Å
    atom_mask: Bool[Array, "N 37"]
    residue_type: Int[Array, " N"]  # unified 147-class vocabulary; -1 outside the decoded tokens
    trajectory: Trajectory


@eqx.filter_jit
def decode(decoder: DecoderTransformer, batch: Batch, trajectory: Trajectory, config: SamplingConfig) -> Design:
    """Decode the final CA + latents to all-atom coordinates and residue types."""
    if config.decode_with_predicted_sequence:
        logits = trajectory.last.sequence_logits
        allowed = decoder.allowed_residue_types[batch.token_type]
        predicted = jnp.argmax(jnp.where(allowed, logits, jnp.finfo(logits.dtype).min), -1)
        fill = batch.latent_mask & (batch.cond_sequence == C.MASK_RESIDUE)
        batch = eqx.tree_at(lambda b: b.cond_sequence, batch, jnp.where(fill, predicted, batch.cond_sequence))
    final = trajectory.final
    out = decoder(batch, DecoderInputs(final.bb_ca, final.local_latents), required_atom_mask=batch.coord_group >= 0)
    no_latent = ~out.emit_mask & batch.padding_mask
    coords = out.coords_nm.at[:, 1].set(jnp.where(no_latent[:, None], final.bb_ca, out.coords_nm[:, 1]))
    return Design(
        coords=coords * 10.0,
        atom_mask=out.atom_mask.at[:, 1].set(out.atom_mask[:, 1] | no_latent),
        residue_type=jnp.where(out.emit_mask, out.residue_type, -1),
        trajectory=trajectory,
    )


class RFD4Proteina(eqx.Module):
    """The flow network, the AE decoder and the trained noise settings."""

    flow: LocalLatentsTransformer
    decoder: DecoderTransformer
    noise_bb_ca: NoiseConfig = eqx.field(static=True)
    noise_local_latents: NoiseConfig = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    @property
    def noise_config(self) -> tuple[NoiseConfig, NoiseConfig]:
        return self.noise_bb_ca, self.noise_local_latents

    def bfloat16(self) -> RFD4Proteina:
        """A copy with bf16 mixed precision in the flow trunk and conditioning (see `layers`). Unlike torch's
        `.bfloat16()` this is not a blanket cast: norms and residual streams compute in float32, and the flow's output
        heads and the decoder keep float32 weights, so the integrator only sees float32 velocities."""
        heads = lambda f: (f.ca_linear, f.local_latents_linear, f.sequence_readout, f.insertion_q_proj,
                           f.insertion_k_proj)
        flow = eqx.tree_at(heads, matmul_dtype(self.flow, jnp.bfloat16), heads(self.flow))
        return eqx.tree_at(lambda m: m.flow, self, flow)


def sample(model: RFD4Proteina, batch: Batch, config: SamplingConfig, *, key: PRNGKeyArray) -> Design:
    """Generate one design for `batch`."""
    noise = draw_noise(key, batch, nsteps=config.nsteps, latent_dim=model.latent_dim)
    trajectory = integrate(model.flow, batch, noise, config, model.noise_config)
    return decode(model.decoder, batch, trajectory, config)


def partial_diffuse(
    model: RFD4Proteina,
    batch: Batch,
    config: SamplingConfig,
    clean: State,
    *,
    key: PRNGKeyArray,
    fraction: float | None = None,
    t_bb_ca: float | None = None,
    t_local_latents: float | None = None,
) -> Design:
    """Re-noise `clean` (e.g. `design.trajectory.final`) to the start of a `partial_schedule` and regenerate.

    Clean latents of an arbitrary structure need the AE encoder, which is not ported.
    """
    sched, start = partial_schedule(config, fraction=fraction, t_bb_ca=t_bb_ca, t_local_latents=t_local_latents)
    noise = draw_noise(key, batch, nsteps=len(sched), latent_dim=model.latent_dim)
    state = corrupt(batch, clean, start, noise, model.noise_config)
    trajectory = simulate(model.flow, batch, state, sched, noise.steps, config, model.noise_config)
    return decode(model.decoder, batch, trajectory, config)


def inverse_fold(
    model: RFD4Proteina, batch: Batch, config: SamplingConfig, ca: Float[Array, "N 3"], *, key: PRNGKeyArray
) -> Design:
    """Generate latents (hence sequence and side chains) for the fixed CA trace `ca` (Å)."""
    noise = draw_noise(key, batch, nsteps=config.nsteps, latent_dim=model.latent_dim)
    state = initial_state(batch, noise, *model.noise_config)
    trajectory = simulate(model.flow, batch, state, schedule(config), noise.steps, config, model.noise_config,
                          fixed_bb_ca=jnp.asarray(ca) / 10.0)
    return decode(model.decoder, batch, trajectory, config)
