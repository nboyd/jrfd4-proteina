"""Torch-free tests (run by default)."""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from jrfd4_proteina.sampling import _schedule, noise_schedule, release_sampling_config, time_schedule


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
    s = _schedule(dataclasses.replace(release_sampling_config()))
    t, stochastic = np.asarray(s.t), np.asarray(s.stochastic)
    assert not stochastic[t > 0.95].any()
    assert not stochastic[t < 0.02].any()
    assert stochastic.any()
