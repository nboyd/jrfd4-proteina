"""Torch-free tests (run by default)."""

import subprocess
import sys

import numpy as np
import pytest

from jrfd4_proteina.sampling import noise_schedule, release_sampling_config, schedule, time_schedule


def test_import_does_not_pull_in_torch():
    code = "import sys, jrfd4_proteina.sampling, jrfd4_proteina.io; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("nsteps", [20, 60, 200])
def test_release_time_schedules_are_increasing_and_end_at_one(nsteps):
    cfg = release_sampling_config()
    for modality in (cfg.bb_ca, cfg.local_latents):
        t = time_schedule(modality, nsteps)
        assert t.shape == (nsteps + 1,)
        assert t.dtype == np.float32
        assert np.all(np.diff(t) > 0)
        assert 0 < t[0] < 2e-3  # TSR-safe start (t=0 is singular for the score)
        assert t[-1] == 1.0
        assert np.all(noise_schedule(modality, t[:-1]) >= 0)


def test_tail_schedule_rejects_too_few_steps():
    cfg = release_sampling_config()
    with pytest.raises(ValueError):
        time_schedule(cfg.bb_ca, 19)  # tail of 10 steps at 2x density needs >= 20


def test_stochastic_band_excludes_both_ends():
    s = schedule(release_sampling_config())
    t, stochastic = np.asarray(s.t), np.asarray(s.stochastic)
    assert not stochastic[t > 0.95].any()
    assert not stochastic[t < 0.02].any()
    assert stochastic.any()


def test_matmul_precision_follows_the_weights():
    import jax
    import jax.numpy as jnp

    from jrfd4_proteina.layers import LayerNorm, Linear, TriangleMultiplication, matmul_dtype

    k = iter(jax.random.split(jax.random.key(0), 16))
    lin = lambda i, o: Linear(jax.random.normal(next(k), (o, i)) / i**0.5, jnp.zeros(o))
    ln = LayerNorm(jnp.ones(8), jnp.zeros(8))
    tri = TriangleMultiplication(ln, ln, *(lin(8, 8) for _ in range(6)), outgoing=True)
    z = jax.random.normal(next(k), (5, 5, 8))
    mask = jnp.ones((5, 5), bool)

    assert lin(8, 4)(z).dtype == jnp.float32
    mixed = matmul_dtype(tri, jnp.bfloat16)
    assert mixed.linear_z.weight.dtype == jnp.bfloat16 and mixed.layer_norm_in.weight.dtype == jnp.float32
    assert mixed.linear_z(z).dtype == jnp.bfloat16  # matmul outputs follow the weights...
    assert mixed.layer_norm_in(z.astype(jnp.bfloat16)).dtype == jnp.float32  # ...norms compute in float32
    full, half = tri(z, mask), mixed(z, mask)
    assert float(jnp.abs(half.astype(jnp.float32) - full).max() / jnp.abs(full).max()) < 0.05
