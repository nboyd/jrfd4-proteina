"""Flow-matching sampler (the reference's `ProductSpaceFlowMatcher.full_simulation`) and decoding.

Two modalities are integrated jointly from t=0 (noise) to t=1 (data): CA coordinates (nm) and local
latents. Each step evaluates the flow network at a recycle depth fixed by a banded schedule; each band
is one `lax.scan`, so the whole sampler is jittable and reverse-mode differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jrfd4_proteina import constants as C
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.features import DecoderInputs, FlowInputs
from jrfd4_proteina.model import (
    DecoderTransformer,
    FlowOutput,
    LocalLatentsTransformer,
    ae_masks,
    flow_emit_masks,
)


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


# --------------------------------------------------------------------------- noise


class Noise(eqx.Module):
    """Every random draw of one sampling run (standard normal / uniform)."""

    bb_ca: Float[Array, "N 3"]
    local_latents: Float[Array, "N L"]
    group_offsets: Float[Array, "G 3"]  # one per centering group, in sorted group order
    steps_bb_ca: Float[Array, "S N 3"]
    steps_local_latents: Float[Array, "S N L"]
    coord_group_priority: Float[Array, "S K"]  # per-call coordinate-group relabelling


def draw_noise(key: PRNGKeyArray, batch: Batch, *, nsteps: int, latent_dim: int) -> Noise:
    n = batch.padding_mask.shape[0]
    keys = jax.random.split(key, 6)
    return Noise(
        bb_ca=jax.random.normal(keys[0], (n, 3)),
        local_latents=jax.random.normal(keys[1], (n, latent_dim)),
        group_offsets=jax.random.normal(keys[2], (n, 3)),
        steps_bb_ca=jax.random.normal(keys[3], (nsteps, n, 3)),
        steps_local_latents=jax.random.normal(keys[4], (nsteps, n, latent_dim)),
        coord_group_priority=jax.random.uniform(keys[5], (nsteps, batch.coord_group.size)),
    )


def masked_mean(x: Float[Array, "N D"], mask: Bool[Array, " N"]) -> Float[Array, " D"]:
    return (x * mask[:, None]).sum(0) / jnp.maximum(mask.sum(), 1)


def initial_state(
    batch: Batch, noise: Noise, bb_ca: NoiseConfig, local_latents: NoiseConfig
) -> tuple[Float[Array, "N 3"], Float[Array, "N L"]]:
    emit_bb, emit_lat = flow_emit_masks(batch)
    x_bb = noise.bb_ca * emit_bb[:, None]
    if bb_ca.zero_com:
        x_bb = (x_bb - masked_mean(x_bb, emit_bb)) * emit_bb[:, None]
    if bb_ca.centering_scale is not None:
        offset_mask = emit_bb & ~batch.coord_cond_mask
        ids = batch.molecule_iid.astype(jnp.int32)
        big = jnp.iinfo(jnp.int32).max
        groups = jnp.unique(jnp.where(offset_mask, ids, big), size=ids.shape[0], fill_value=big)
        offset = noise.group_offsets[jnp.searchsorted(groups, ids)] * offset_mask[:, None]
        x_bb = (x_bb + bb_ca.centering_scale * offset) * emit_bb[:, None]
    if local_latents.zero_com or local_latents.centering_scale is not None:
        raise NotImplementedError("latent centering is not used by the release")
    return x_bb, noise.local_latents * emit_lat[:, None]


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


class _Schedule(eqx.Module):
    t: Float[Array, "S 2"]
    t_next: Float[Array, "S 2"]
    g: Float[Array, "S 2"]
    stochastic: Bool[Array, "S 2"]


def _schedule(config: SamplingConfig) -> _Schedule:
    ts, gs, stoch = [], [], []
    for cfg in (config.bb_ca, config.local_latents):
        t = time_schedule(cfg, config.nsteps)
        ts.append(t)
        gs.append(noise_schedule(cfg, t[:-1]))
        stoch.append((t[:-1] <= cfg.t_lim_ode) & (t[:-1] >= cfg.t_lim_ode_below) & (cfg.sampling_mode == "vf_tsr"))
    t = np.stack(ts, -1)
    return _Schedule(t=t[:-1], t_next=t[1:], g=np.stack(gs, -1), stochastic=np.stack(stoch, -1))


class Trajectory(eqx.Module):
    x_bb_ca: Float[Array, "N 3"]
    x_local_latents: Float[Array, "N L"]
    last: FlowOutput  # network output at the final step


@eqx.filter_jit
def integrate(
    model: LocalLatentsTransformer,
    batch: Batch,
    noise: Noise,
    config: SamplingConfig,
    noise_config: tuple[NoiseConfig, NoiseConfig],
) -> Trajectory:
    """Run the sampler from given draws (deterministic)."""
    widths = [w for w, _ in config.recycle_schedule]
    if sum(widths) != config.nsteps:
        raise ValueError(f"recycle schedule covers {sum(widths)} steps, expected {config.nsteps}")
    emit_bb, emit_lat = flow_emit_masks(batch)
    sched = _schedule(config)
    x = initial_state(batch, noise, *noise_config)
    last = None
    start = 0
    for width, depth in config.recycle_schedule:
        band = slice(start, start + width)
        xs = (
            jax.tree.map(lambda a, band=band: a[band], sched),
            noise.steps_bb_ca[band], noise.steps_local_latents[band], noise.coord_group_priority[band],
        )

        def body(x, step, depth=depth):
            s, eps_bb, eps_lat, priority = step
            inputs = FlowInputs(x[0], x[1], s.t[0], s.t[1], priority)
            out = model(batch, inputs, depth)
            x_bb = _step(config.bb_ca, x[0], out.v_bb_ca, s.t[0], s.t_next[0], s.g[0], s.stochastic[0], eps_bb,
                         emit_bb)
            x_lat = _step(config.local_latents, x[1], out.v_local_latents, s.t[1], s.t_next[1], s.g[1],
                          s.stochastic[1], eps_lat, emit_lat)
            return (x_bb, x_lat), out

        x, outs = jax.lax.scan(body, x, xs)
        last = jax.tree.map(lambda a: a[-1], outs)
        start += width
    return Trajectory(x[0], x[1], last)


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
        _, emit = ae_masks(batch)
        fill = emit & (batch.cond_sequence == C.MASK_RESIDUE)
        batch = eqx.tree_at(lambda b: b.cond_sequence, batch, jnp.where(fill, predicted, batch.cond_sequence))
    out = decoder(
        batch, DecoderInputs(trajectory.x_bb_ca, trajectory.x_local_latents), required_atom_mask=batch.coord_group >= 0
    )
    no_latent = ~out.emit_mask & batch.padding_mask
    coords = out.coords_nm.at[:, 1].set(jnp.where(no_latent[:, None], trajectory.x_bb_ca, out.coords_nm[:, 1]))
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


def sample(model: RFD4Proteina, batch: Batch, config: SamplingConfig, *, key: PRNGKeyArray) -> Design:
    """Generate one design for `batch`."""
    noise = draw_noise(key, batch, nsteps=config.nsteps, latent_dim=model.latent_dim)
    trajectory = integrate(model.flow, batch, noise, config, (model.noise_bb_ca, model.noise_local_latents))
    return decode(model.decoder, batch, trajectory, config)
