"""Muon, as the reference fine-tunes with it (`torch.optim.Muon`, `adjust_lr_fn="match_rms_adamw"`), for optax.

    optimizer = optim.optimizer(5e-5)
    opt_state = optimizer.init(adapters)

Every parameter must be a matrix or a stack of matrices: each `[..., M, N]` slice is updated on its own. That
covers LoRA adapters; biases, norms and embeddings would need Adam, as in the reference.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float


def orthogonalize(
    g: Float[Array, "*B M N"], coefficients: tuple[float, float, float], steps: int, eps: float
) -> Float[Array, "*B M N"]:
    """Newton-Schulz iteration towards the nearest semi-orthogonal matrix (`U V^T` of the SVD)."""
    a, b, c = coefficients
    wide = g.shape[-2] <= g.shape[-1]
    x = g if wide else g.mT
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=(-2, -1), keepdims=True), eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x if wide else x.mT


def scale_by_muon(
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> optax.GradientTransformation:
    """Orthogonalized momentum, scaled by `0.2 * sqrt(max(M, N))` so its RMS matches an AdamW step."""

    def init(params):
        if any(p.ndim < 2 for p in jax.tree.leaves(params)):
            raise ValueError("Muon updates matrices; every parameter needs at least two dimensions")
        return jax.tree.map(jnp.zeros_like, params)

    def update(grads, buffers, params=None):
        buffers = jax.tree.map(lambda buf, g: buf + (1.0 - momentum) * (g - buf), buffers, grads)

        def direction(g, buf):
            u = g + momentum * (buf - g) if nesterov else buf
            return 0.2 * math.sqrt(max(g.shape[-2:])) * orthogonalize(u, ns_coefficients, ns_steps, eps)

        return jax.tree.map(direction, grads, buffers), buffers

    return optax.GradientTransformation(init, update)


def muon(learning_rate: optax.ScalarOrSchedule, weight_decay: float = 0.1, **kwargs) -> optax.GradientTransformation:
    """`p <- (1 - lr * weight_decay) p - lr * scale_by_muon(grad)`."""
    return optax.chain(
        scale_by_muon(**kwargs), optax.add_decayed_weights(weight_decay), optax.scale_by_learning_rate(learning_rate)
    )


def optimizer(
    learning_rate: optax.ScalarOrSchedule = 5e-5, *, max_grad_norm: float = 1.0, accumulate: int = 1
) -> optax.GradientTransformation:
    """The reference's fine-tuning optimizer: gradient-norm clipping, then Muon. `accumulate > 1` averages the
    gradients of that many `train_step`s into one update; updates with non-finite gradients are skipped."""
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), muon(learning_rate))
    tx = optax.apply_if_finite(tx, max_consecutive_errors=10)
    return optax.MultiSteps(tx, accumulate).gradient_transformation() if accumulate > 1 else tx
