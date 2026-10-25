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

batch = convert.featurize(spec)[0]                # an RFD4 condition spec as a dict, {example_id: spec}
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

### Precision and speed

`model.bfloat16()` returns a copy that runs the flow trunk in bf16 mixed precision, as the reference's `bf16-mixed`
default does: matmuls take bf16 inputs and accumulate in float32, norms and softmax statistics run in float32, residual
streams stay float32, and the flow's output heads and the decoder are untouched, so the integrator only sees float32
velocities. Precision follows the weights' dtype, so the float32 model is unaffected.

```python
fast = model.bfloat16()  # ~1.6x faster batched sampling
```

Per-design sampling time, PD-L1 with a 60-residue binder (175 tokens), one H100, steady state:

| | batch 1 | batch 16 |
|---|---|---|
| this port, `model.bfloat16()` | 1.7 s | 1.2 s |
| this port, float32 (TF32 matmuls, the JAX default) | 2.3 s | 1.9 s |
| reference `rfd4-generate` defaults (`bf16-mixed`, `torch.compile`; includes featurization and I/O, pads to 192 tokens) | 3.8 s | 1.0-1.1 s |
| reference float32 eager (TF32) | 6.1-6.4 s | 3.3-3.7 s |

bf16 changes single network evaluations by 1-8% (TF32: under 1%), but 256 bf16 PD-L1 designs are statistically
indistinguishable from the reference's float32 designs (interface, hotspot, geometry and clash metrics: KS p >= 0.42;
residue types: Jensen-Shannon distance 0.016, chi-squared p 0.76).

### API

| | |
|---|---|
| `sample(model, batch, config, *, key) -> Design` | draw noise, integrate, decode |
| `integrate(flow, batch, noise, config, noise_config) -> Trajectory` | deterministic sampler given `Noise` (reverse-mode differentiable) |
| `simulate(flow, batch, state, schedule, steps, config, noise_config, fixed_bb_ca=None) -> Trajectory` | integrate a `State` through any slice of a `Schedule` |
| `schedule(config)` / `partial_schedule(config, *, fraction \| t_bb_ca, t_local_latents)` | per-step times, noise scales and recycle depths; slice with `[a:b]` |
| `partial_diffuse(model, batch, config, clean, *, key, ...)` / `corrupt(...)` | partial diffusion from a clean `State` (e.g. `design.trajectory.final`) |
| `inverse_fold(model, batch, config, ca, *, key)` | generate latents/sequence for a fixed CA trace (Å) |
| `search.beam_search(model, batch, config, checkpoints, *, key, ...)` | reward-guided beam search |
| `decode(decoder, batch, trajectory, config) -> Design` | AE decode, conditioned on the predicted sequence |
| `LocalLatentsTransformer.__call__(batch, FlowInputs, n_recycle) -> FlowOutput` | one network evaluation (velocities, sequence and insertion logits) |
| `DecoderTransformer.__call__(batch, DecoderInputs, required_atom_mask=None) -> DecoderOutput` | |
| `RFD4Proteina.bfloat16() -> RFD4Proteina` | bf16 mixed-precision flow trunk (see above); `layers.matmul_dtype` for any module |
| `SamplingConfig` / `release_sampling_config()` | schedules, TSR temperature, recycle bands (reference defaults: 60 steps, medium front-loaded recycling) |
| `Batch` | typed design inputs; `Batch.from_reference` adapts the reference collated batch |
| `convert.load / convert_model / featurize / batch_from_reference` | torch-dependent conversion |
| `io.save / io.load` | torch-free serialization |

### Example: PD-L1 binder

The reference's ["design a binder in 10 lines"](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/how-to/binder-in-10-lines.md)
spec (also `tests/data/pdl1_binder.yaml`): a 50-120 residue binder against PD-L1 given the target's backbone
geometry, sequence and three hotspots.

```python
spec = {"pdl1_binder": {
    "input": "tests/data/5o45_cropped.cif.zst",   # relative to the working directory
    "contig": "50-120, /0, A17-131",
    "conditions": {
        "C_CRDGRP": {1: [{"select": "sel('A/*/*/[N,CA,C,O]')"}]},
        "C_SEQ": {True: [{"select": "sel('A')"}]},
        "C_HOT": {True: [{"select": s} for s in ("sel('A/*/56/[CG,OH]')", "sel('A/*/115/[CG,SD]')",
                                                  "sel('A/*/123/[CD2,OH]')")]},
    },
}}
batch = convert.featurize(spec, seed=0)[0]       # the binder length is sampled from the contig; seed fixes it
design = sample(model, batch, release_sampling_config(), key=jax.random.key(0))
```

A `Batch` holds one sampled layout (one binder length): featurize again, with other seeds or fixed lengths
(`"contig": "80, /0, A17-131"`), to cover a range. Featurizing takes ~0.3 s after the first call.

### Beam search

At each checkpoint every survivor branches `n_branch` times, the branches are integrated to the checkpoint with
fresh noise and scored (rolled out to t=1 and decoded, or with `use_clean_prediction` decoded from their last
clean prediction), and the best `beam_width` continue. Rewards are maximized and run inside a jitted map over
candidates: write them as `eqx.Module`s holding their models as fields, so they are passed, not closed over.

```python
from jrfd4_proteina.search import Checkpoint, beam_search

class Compact(eqx.Module):                        # any JAX reward: (design, *, key) -> reward | (reward, aux)
    def __call__(self, design, *, key):
        ca, m = design.coords[:, 1], design.atom_mask[:, 1]
        center = (ca * m[:, None]).sum(0) / m.sum()
        return -jnp.sqrt((((ca - center) ** 2).sum(-1) * m).sum() / m.sum())

result = beam_search(model, batch, config, [Checkpoint(20, Compact()), Checkpoint(40, Compact()),
                                            Checkpoint(60, Compact())], key=jax.random.key(0), beam_width=4, n_branch=4)
result.designs, result.rewards                    # final beam [4], best first; result.checkpoints: every candidate
```

Search steps the sampler one step at a time (one compile per recycle depth), so checkpoints can sit anywhere.

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

* Partial diffusion (`partial_trajectory_fraction_steps`, explicit and coupled `partial_t_*`), inverse folding
  (`resampled_forward_marginal`) and beam search (`BeamSearchStrategy`: per-checkpoint rewards and branching,
  rollout or clean-prediction scoring, continuation criteria, resuming from saved states).

Not ported: training, the AE encoder (so partial diffusion starts from latents this model produced, not from an
arbitrary structure), classifier-free guidance (unused by the release checkpoint), differentiable-metric
guidance, metrics and refolding oracles. Featurization is delegated to the reference
pipeline (`convert.featurize`, needs torch; condition specs only, not condition-annotated CIFs); a torch-free
featurizer is not included.

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
