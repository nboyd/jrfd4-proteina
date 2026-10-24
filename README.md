# jrfd4-proteina

A JAX/Equinox port of the inference path of [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina)
(Baker Lab × NVIDIA GenAIR): the flow network, the flow-matching sampler and the autoencoder decoder.
Everything is single-example, pure and jittable; batch with `jax.vmap`.

RFD4-Proteina is a **flow-matching** model (Proteina lineage), not an AF3-style EDM diffusion model. It
integrates two modalities jointly from noise (t=0) to data (t=1): per-token CA coordinates (nm) and
32-dimensional per-token local latents. A transformer with pair-biased attention and triangle updates
predicts their velocities, recycling its trunk a schedule-dependent number of times per step, and an
autoencoder decoder maps the final CA + latents to residue types and atom37 coordinates.

## Install

```bash
uv sync                         # core (jax, equinox): torch-free
uv sync --extra convert         # + torch and rfd4-proteina, for weight conversion and featurization
uv sync --extra cuda            # + jax[cuda12]
```

## Use

```python
import jax
from jrfd4_proteina import convert, io
from jrfd4_proteina.sampling import release_sampling_config, sample

model = convert.load("nvidia/RFD4-Proteina")      # converts the released bundle (needs the convert extra)
io.save(model, "rfd4_proteina")                   # later: io.load("rfd4_proteina"), torch-free

batch = convert.featurize("design.yaml")[0]       # any RFD4 condition spec / annotated CIF
design = sample(model, batch, release_sampling_config(), key=jax.random.key(0))
design.coords, design.atom_mask, design.residue_type  # Å, [N, 37]; unified 147-class vocabulary
```

Several designs at once: pass the model as an argument (never close over it in a jitted function):

```python
import equinox as eqx
config, keys = release_sampling_config(), jax.random.split(jax.random.key(0), 8)
designs = eqx.filter_jit(eqx.filter_vmap(lambda m, b, k: sample(m, b, config, key=k), in_axes=(None, None, 0)))(
    model, batch, keys)
```

### API

| | |
|---|---|
| `sample(model, batch, config, *, key) -> Design` | draw noise, integrate, decode |
| `integrate(flow, batch, noise, config, noise_config) -> Trajectory` | deterministic sampler given `Noise` (reverse-mode differentiable) |
| `decode(decoder, batch, trajectory, config) -> Design` | AE decode, conditioned on the predicted sequence |
| `LocalLatentsTransformer.__call__(batch, FlowInputs, n_recycle) -> FlowOutput` | one network evaluation (velocities, sequence and insertion logits) |
| `DecoderTransformer.__call__(batch, DecoderInputs, required_atom_mask=None) -> DecoderOutput` | |
| `SamplingConfig` / `release_sampling_config()` | schedules, TSR temperature, recycle bands (reference defaults: 60 steps, medium front-loaded recycling) |
| `Batch` | typed design inputs; `Batch.from_reference` adapts the reference collated batch |
| `convert.load / convert_model / featurize / batch_from_reference` | torch-dependent conversion |
| `io.save / io.load` | torch-free serialization |

Each recycle band of the schedule is one `lax.scan` with a static trunk depth; within a step, early recycles run
under `stop_gradient` (as in the reference) and the final one is differentiated.

## What is ported

* Flow network (`LocalLatentsTransformer`): all 21 sequence, 4 conditioning and 10 pair features the release uses,
  including coordinate/annotation conditioning, chirality vectors with noise alignment, and per-call random
  relabelling of coordinate groups; pair-biased attention with adaptive norms, triangle multiplicative
  updates, recycling with remaining-pass conditioning, sequence and insertion heads.
* Sampler: TSR-safe log/power schedules with dense tails, `1/t` and `tan` noise schedules, the `vf_tsr`
  step (temporal score rescaling, stochastic in the middle band, deterministic near both ends) and the plain
  `vf` ODE; zero-COM initial noise with per-molecule stochastic centering; banded recycle depths.
* AE decoder (`DecoderTransformer`) and sequence-conditioned decoding.

Not ported: training, the AE encoder, partial diffusion, inverse folding, classifier-free guidance (unused by
the release checkpoint), reward-guided/beam search, metrics. Featurization is delegated to the reference
pipeline (`convert.featurize`, needs torch); a torch-free featurizer is not included.

## Parity with the PyTorch reference

CPU, fp32 (`highest` matmul precision), released EMA weights (HF revision `48377e4`), 100-residue monomer
benchmark. Relative errors are `max|Δ| / max|reference|`.

| Level | Result |
|---|---|
| Leaves and layers (norms, attention, transitions, triangle updates, pair update) | 1e-7 – 1e-6 |
| Every feature creator and factory | exact, except time embeddings (below) |
| Full network forward, recycle depth 2 (velocities, sequence logits) | 5e-7 – 1e-6; insertion logits 2e-6 |
| AE decoder | 2e-5; residue types and atom masks identical |
| Time / noise schedules | 1 ulp |
| End-to-end sampling with replayed noise, plain ODE (24 steps, 42 trunk passes) | CA 2e-5, decoded atoms < 0.012 Å, residue types identical |
| End-to-end sampling with replayed noise, release TSR sampler | identical noise consumption; every step agrees to ~1e-6 from the reference state |

Two fp32 floors, not port errors:

* Time embeddings evaluate `sin`/`cos` at up to ~1e3 rad, where one float32 ulp is 1.2e-4; their
  high-frequency channels differ by up to 5e-5 from torch (torch on GPU would differ similarly). The
  projection right after agrees to 4e-7.
* Decoder attention logits reach ~1e3: torch and this port are equally far (9e-6) from a float64 evaluation.

The release TSR sampler amplifies 1e-6 perturbations of its initial state about 1e4-fold, so free-running
TSR trajectories of the two implementations separate even though each step agrees; end-to-end numeric
parity is therefore asserted under the (stable) plain-ODE sampler.

Run the checks: `uv run pytest` (torch-free) and `uv run --extra convert pytest -m reference` (downloads the
checkpoint; about 3 minutes on CPU).

## Attribution

RFD4-Proteina code and weights: Copyright (c) 2026, Institute for Protein Design, University of Washington,
BSD 3-Clause (see [NOTICE](NOTICE)). This port is BSD 3-Clause ([LICENSE](LICENSE)). No weights are
redistributed here.
