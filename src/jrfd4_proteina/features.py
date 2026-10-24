"""Input features of the flow network and the AE decoder, for a single example.

Each feature maps `(batch, inputs) -> array` of shape `[d]` (per example), `[N, d]` (per token) or
`[N, N, d]` (per pair); a `FeatureFactory` concatenates its features in checkpoint order and projects
them. `inputs` carries the per-call state (noisy sample, times, ...).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from jrfd4_proteina import constants as C
from jrfd4_proteina.backend import from_torch, register_from_torch
from jrfd4_proteina.batch import Batch
from jrfd4_proteina.layers import Embedding, Linear, Norm


class FlowInputs(eqx.Module):
    """Per-call inputs of the flow network."""

    x_bb_ca: Float[Array, "N 3"]  # noisy CA coordinates, nm
    x_local_latents: Float[Array, "N L"]
    t_bb_ca: Float[Array, ""]
    t_local_latents: Float[Array, ""]
    # Uniform draws that fix this call's random relabelling of coordinate groups (`CoordGroup`).
    coord_group_priority: Float[Array, " K"]
    recycle_remaining: Float[Array, " R"] | None = None  # one-hot remaining passes; None before the first


class DecoderInputs(eqx.Module):
    ca_coors_nm: Float[Array, "N 3"]
    z_latent: Float[Array, "N L"]


def time_embedding(t: Float[Array, ""], dim: int, max_positions: int = 2000) -> Float[Array, " D"]:
    half = dim // 2
    # Correctly rounded constants: angles reach ~1e3 rad, where a 1-ulp frequency error is visible.
    freqs = np.exp(np.arange(half) * -(math.log(max_positions) / (half - 1))).astype(np.float32)
    angles = (t * max_positions) * freqs
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])


def one_hot_bins(x: Float[Array, ...], limits: Float[Array, " B"]) -> Float[Array, "... B1"]:
    """`torch.bucketize` (right-closed bins) followed by one-hot over `len(limits) + 1` bins."""
    return jax.nn.one_hot(jnp.searchsorted(limits, x, side="left"), limits.shape[0] + 1)


def proper_rotation(cov: Float[Array, "... 3 3"]) -> Float[Array, "... 3 3"]:
    """Rotation `R` minimising `|source @ R - target|` given `cov = source.T @ target` (Kabsch)."""
    u, _, vh = jnp.linalg.svd(cov)
    signs = jnp.ones(cov.shape[:-1]).at[..., -1].set(jnp.sign(jnp.linalg.det(u @ vh)))
    return (u * signs[..., None, :]) @ vh


# --------------------------------------------------------------------------- sampling state


class XtBBCA(eqx.Module):
    def __call__(self, batch: Batch, inputs: FlowInputs):
        return inputs.x_bb_ca


class XtLocalLatents(eqx.Module):
    def __call__(self, batch: Batch, inputs: FlowInputs):
        return inputs.x_local_latents


class TimeEmbedding(eqx.Module):
    dim: int = eqx.field(static=True)
    modality: str = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs: FlowInputs):
        return time_embedding(getattr(inputs, f"t_{self.modality}"), self.dim)


class RecycleRemaining(eqx.Module):
    dim: int = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs: FlowInputs):
        return jnp.zeros(self.dim) if inputs.recycle_remaining is None else inputs.recycle_remaining


class CaCoors(eqx.Module):
    def __call__(self, batch: Batch, inputs: DecoderInputs):
        return inputs.ca_coors_nm


class Latent(eqx.Module):
    def __call__(self, batch: Batch, inputs: DecoderInputs):
        return inputs.z_latent


# --------------------------------------------------------------------------- token identity


class TokenType(eqx.Module):
    embedding: Embedding

    def __call__(self, batch: Batch, inputs):
        return self.embedding(batch.token_type)


class CondResidueType(eqx.Module):
    embedding: Embedding  # identity in the release; kept as weights for fidelity

    def __call__(self, batch: Batch, inputs):
        conditioned = batch.cond_sequence != C.MASK_RESIDUE
        return self.embedding(batch.cond_sequence * conditioned) * conditioned[:, None]


class SequenceProbabilities(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        p = batch.sequence_probabilities
        specified = ~jnp.isnan(p)
        return jnp.concatenate([jnp.where(specified, p, 0.0), specified.astype(jnp.float32)], -1)


def _atomized(batch: Batch) -> Float[Array, "N 1"]:
    return (batch.token_type == C.TOKEN_TYPE_ATOMIZED)[:, None]


class AtomCharge(eqx.Module):
    """Charge of atomized tokens (atom37 slot 1), embedded over the saturated vocabulary -4..+6."""

    embedding: Embedding

    def __call__(self, batch: Batch, inputs):
        index = jnp.clip(batch.charge[:, 1].astype(jnp.int32) + 4, 0, 10)
        return self.embedding(index) * _atomized(batch)


class AtomicElement(eqx.Module):
    embedding: Embedding

    def __call__(self, batch: Batch, inputs):
        index = jnp.clip(batch.element[:, 1].astype(jnp.int32), 0, self.embedding.weight.shape[0] - 1)
        return self.embedding(index) * _atomized(batch)


class CathCode(eqx.Module):
    embedding_C: Embedding
    embedding_A: Embedding
    embedding_T: Embedding

    def __call__(self, batch: Batch, inputs):
        codes = batch.cath.astype(jnp.int32) + 1  # -1 (absent) -> null class 0
        return jnp.concatenate(
            [self.embedding_C(codes[:, 0]), self.embedding_A(codes[:, 1]), self.embedding_T(codes[:, 2])], -1
        )


# --------------------------------------------------------------------------- atom37 annotations


def _on_atoms(x, batch: Batch):
    return x.astype(jnp.float32) * batch.atom37_mask


class CondCoords(eqx.Module):
    """Absolute coordinate conditioning: flattened coordinates (nm) and their mask."""

    def __call__(self, batch: Batch, inputs):
        coords = batch.cond_coords
        mask = jnp.isfinite(coords).any(-1)
        coords = jnp.nan_to_num(coords, nan=0.0) / 10.0 * mask[..., None]
        return jnp.concatenate([coords.reshape(coords.shape[0], -1), mask.astype(jnp.float32)], -1)


class CondInternalCoords(eqx.Module):
    """Internal coordinate conditioning (groups other than unknown / absolute)."""

    def __call__(self, batch: Batch, inputs):
        mask = (batch.coord_group != C.UNKNOWN_COORDS) & (batch.coord_group != C.KNOWN_ABSOLUTE_COORDS)
        coords = jnp.nan_to_num(batch.cond_internal_coords) / 10.0 * mask[..., None]
        return jnp.concatenate([coords.reshape(coords.shape[0], -1), mask.astype(jnp.float32)], -1)


class Rasa(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        return jnp.where(batch.atom37_mask, batch.rasa, C.RASA_UNSPECIFIED).astype(jnp.float32)


class SecondaryStructure(eqx.Module):
    ss_embedding: Embedding

    def __call__(self, batch: Batch, inputs):
        return self.ss_embedding(batch.secondary_structure.astype(jnp.int32))


class HBond(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        hb = batch.hbond
        donor = (hb == C.HBOND_DONOR) | (hb == C.HBOND_BOTH)
        acceptor = (hb == C.HBOND_ACCEPTOR) | (hb == C.HBOND_BOTH)
        return jnp.concatenate([_on_atoms(donor, batch), _on_atoms(acceptor, batch)], -1)


class PiCation(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        pc = batch.pication
        aromatic = (pc == C.PICATION_AROMATIC) | (pc == C.PICATION_BOTH)
        cation = (pc == C.PICATION_CATION) | (pc == C.PICATION_BOTH)
        return jnp.concatenate([_on_atoms(aromatic, batch), _on_atoms(cation, batch)], -1)


class AtomFlag(eqx.Module):
    """A per-atom annotation, as `annotation != 0` on resolved atoms (pistack, salt bridge, ...)."""

    field: str = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs):
        return _on_atoms(getattr(batch, self.field) != 0, batch)


class NaSsState(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        return jnp.stack([batch.na_ss_state, batch.na_ss_state_mask], -1).astype(jnp.float32)


class ChainCentroid(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        c = batch.chain_centroid
        return jnp.where(jnp.isfinite(c).all(-1, keepdims=True), c / 10.0, 0.0)


class CoordGroup(eqx.Module):
    """Coordinate group ids, randomly relabelled per call so the model cannot read meaning into them.

    The relabelling is the permutation ranking `inputs.coord_group_priority` over the distinct groups.
    """

    def __call__(self, batch: Batch, inputs: FlowInputs):
        groups = batch.coord_group.astype(jnp.int32)
        present = groups != C.UNKNOWN_COORDS
        k = groups.size
        big = jnp.iinfo(jnp.int32).max
        distinct = jnp.unique(jnp.where(present, groups, big).ravel(), size=k, fill_value=big)
        n_distinct = (distinct != big).sum()
        rank = jnp.searchsorted(distinct, groups)
        priority = jnp.where(jnp.arange(k) < n_distinct, inputs.coord_group_priority, jnp.inf)
        relabel = jnp.argsort(jnp.argsort(priority))
        return jnp.where(present, relabel[rank], groups).astype(jnp.float32)


class Chirality(eqx.Module):
    """Packed chirality vectors, each rotated onto its frame's current noisy neighbour directions."""

    noise_align: bool = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs: FlowInputs):
        vectors = batch.chirality.astype(jnp.float32)
        if not self.noise_align:
            return vectors
        return align_chirality(vectors, batch, jax.lax.stop_gradient(inputs.x_bb_ca))


def align_chirality(
    vectors: Float[Array, "N S3"], batch: Batch, x_bb_ca: Float[Array, "N 3"]
) -> Float[Array, "N S3"]:
    n = vectors.shape[0]
    packed = vectors.reshape(n, -1, 3)  # [N, S, 3]
    n_slots = packed.shape[1]
    frames = batch.chiral_frame_atoms.astype(jnp.int32)  # [N, S, 2 + K]
    valid = batch.padding_mask
    active = (frames[..., 0] >= 0) & valid[:, None]  # [N, S]
    present = frames[..., 2:] >= 0  # [N, S, K]
    neighbours = jnp.maximum(frames[..., 2:], 0)
    supplied = packed[neighbours, jnp.arange(n_slots)[None, :, None]]  # [N, S, K, 3]
    origin = x_bb_ca[frames[..., :2]].mean(-2)  # [N, S, 3]
    noisy = x_bb_ca[neighbours] - origin[:, :, None]  # [N, S, K, 3]
    supplied = jnp.where(present[..., None], supplied, 0.0)
    noisy = jnp.where(present[..., None], noisy, 0.0)
    noisy = noisy / jnp.maximum(jnp.linalg.norm(noisy, axis=-1, keepdims=True), 1e-8)
    rotation = proper_rotation(jnp.einsum("nskx,nsky->nsxy", supplied, noisy))
    rotation = jnp.where(active[..., None, None], rotation, jnp.eye(3)).reshape(n * n_slots, 3, 3)
    owners = batch.chiral_slot_owner.astype(jnp.int32)  # flattened (token, slot) frame owning each slot
    aligned = jnp.einsum("nsx,nsxy->nsy", packed, rotation[jnp.maximum(owners, 0)])
    aligned = jnp.where((owners >= 0)[..., None], aligned, packed)
    return aligned.reshape(n, -1) * valid[:, None]


# --------------------------------------------------------------------------- pair features


class XtPairDistances(eqx.Module):
    min_dist: float = eqx.field(static=True)
    max_dist: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    from_decoder: bool = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs):
        x = inputs.ca_coors_nm if self.from_decoder else inputs.x_bb_ca
        d = jnp.linalg.norm(x[:, None] - x[None, :], axis=-1)
        return one_hot_bins(d, jnp.linspace(self.min_dist, self.max_dist, self.dim - 1))


class PairFlag(eqx.Module):
    """A per-pair annotation as one float channel."""

    field: str = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs):
        return getattr(batch, self.field)[..., None].astype(jnp.float32)


class SameToken(eqx.Module):
    """1 where two tokens share a token-level label (index group, molecule, entity)."""

    field: str = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs):
        label = getattr(batch, self.field)
        return (label[:, None] == label[None, :])[..., None].astype(jnp.float32)


class BondOrder(eqx.Module):
    dim: int = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs):
        return jax.nn.one_hot(batch.bond_order.astype(jnp.int32), self.dim)


class PathLength(eqx.Module):
    dim: int = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs):
        return jax.nn.one_hot(jnp.minimum(batch.token_bond_path_length.astype(jnp.int32), self.dim - 1), self.dim)


class NaSsPair(eqx.Module):
    def __call__(self, batch: Batch, inputs):
        return jnp.stack([batch.na_ss_pair, batch.na_ss_pair_mask], -1).astype(jnp.float32)


# --------------------------------------------------------------------------- factories


class FeatureFactory(eqx.Module):
    """Concatenate features, mask padding, project (and optionally normalise), mask again.

    `mode` is "seq" ([N, d]), "pair" ([N, N, d]) or "example" (all features per example, output [d]).
    """

    features: tuple[eqx.Module, ...]
    linear_out: Linear
    ln_out: Norm | None
    mode: str = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs) -> Float[Array, ...]:
        mask = batch.padding_mask
        n = mask.shape[0]
        outputs = [f(batch, inputs) for f in self.features]
        if self.mode == "example":
            x = jnp.concatenate(outputs, -1) * mask.any()
        else:
            spatial = (n,) if self.mode == "seq" else (n, n)
            outputs = [jnp.broadcast_to(o, spatial + o.shape[-1:]) if o.ndim == 1 else o for o in outputs]
            x = _mask(jnp.concatenate(outputs, -1), mask, self.mode)
        x = self.linear_out(x)
        if self.ln_out is not None:
            x = self.ln_out(x)
        return x * mask.any() if self.mode == "example" else _mask(x, mask, self.mode)


def _mask(x, mask: Bool[Array, " N"], mode: str):
    if mode == "seq":
        return x * mask[:, None]
    return x * (mask[:, None] & mask[None, :])[..., None]


class ZeroFeatures(eqx.Module):
    """The reference's empty factory: zeros of a fixed width, unprojected."""

    dim: int = eqx.field(static=True)

    def __call__(self, batch: Batch, inputs) -> Float[Array, "N D"]:
        return jnp.zeros((batch.padding_mask.shape[0], self.dim))


# --------------------------------------------------------------------------- conversion

_FF = "rfproteina.nn.feature_factory"


def _convert_feature(m) -> eqx.Module:
    kind = type(m).__name__
    simple = {
        "XtBBCASeqFeat": XtBBCA, "XtLocalLatentsSeqFeat": XtLocalLatents, "CaCoorsNanometersSeqFeat": CaCoors,
        "LatentVariableSeqFeat": Latent, "SequenceProbabilitiesSeqFeat": SequenceProbabilities,
        "ConditionalAtom37NanometersCoorsSeqFeat": CondCoords, "RasaBinnedSeqFeat": Rasa,
        "ModernizedHydrogenBondSeqFeat": HBond, "PiCationSeqFeat": PiCation, "ChainCentroidXyzSeqFeat": ChainCentroid,
        "NucleicSecondaryStructureStateSeqFeat": NaSsState, "CoordGroupSeqFeat": CoordGroup,
        "NucleicSecondaryStructurePairFeat": NaSsPair,
    }
    flags = {
        "PiStackingSeqFeat": "pistack", "SaltBridgeSeqFeat": "saltbridge", "HydrophobicSeqFeat": "hydrophobic",
        "MetalCoordinationSeqFeat": "metal_coord", "HotspotSeqFeat": "hotspot",
    }
    pair_flags = {"BondMaskPairFeat": "bond_mask", "IsAromaticPairFeat": "is_aromatic",
                  "CanonicalPolymerOrderingPairFeat": "canonical_polymer_ordering"}
    same = {"IndexGroupPairFeat": "index_group", "MoleculeIdxPairFeat": "molecule_iid",
            "MoleculeEntityPairFeat": "molecule_entity_id"}
    if kind in simple:
        return simple[kind]()
    if kind in flags:
        return AtomFlag(field=flags[kind])
    if kind in pair_flags:
        return PairFlag(field=pair_flags[kind])
    if kind in same:
        return SameToken(field=same[kind])
    if kind == "ConditionalAtom37NanometersInternalCoorsSeqFeat":
        if m.noise_align or m.translate:
            raise NotImplementedError("internal-coordinate noise alignment is not ported")
        return CondInternalCoords()
    if kind == "TimeEmbeddingBatchFeat":
        return TimeEmbedding(dim=m.dim, modality=m.data_mode_use)
    if kind == "RecycleRemainingBatchFeat":
        return RecycleRemaining(dim=m.dim)
    if kind == "TokenTypeSeqFeat":
        return TokenType(embedding=from_torch(m.embedding))
    if kind == "ConditionalResidueTypeSeqFeat":
        return CondResidueType(embedding=from_torch(m.embedding))
    if kind == "AtomChargeSeqFeat":
        return AtomCharge(embedding=from_torch(m.embedding))
    if kind == "AtomicElementSeqFeat":
        return AtomicElement(embedding=from_torch(m.embedding))
    if kind == "FoldEmbeddingSeqFeat":
        return CathCode(*(from_torch(getattr(m, f"embedding_{c}")) for c in "CAT"))
    if kind == "SecondaryStructureSeqFeat":
        return SecondaryStructure(ss_embedding=from_torch(m.ss_embedding))
    if kind == "ChiralitySeqFeat":
        if m.translate:
            raise NotImplementedError("chirality translation is not ported")
        return Chirality(noise_align=m.noise_align)
    if kind == "XtBBCAPairwiseDistancesPairFeat":
        return XtPairDistances(min_dist=m.min_dist, max_dist=m.max_dist, dim=m.dim, from_decoder=False)
    if kind == "CaCoorsNanometersPairwiseDistancesPairFeat":
        return XtPairDistances(min_dist=m.min_dist, max_dist=m.max_dist, dim=m.dim, from_decoder=True)
    if kind == "BondOrderPairFeat":
        return BondOrder(dim=m.dim)
    if kind == "TokenBondPathLengthPairFeat":
        return PathLength(dim=m.dim)
    raise NotImplementedError(f"feature {kind} is not ported")


@register_from_torch(f"{_FF}.FeatureFactory")
def _convert_factory(m) -> FeatureFactory | ZeroFeatures:
    if m.ret_zero:
        return ZeroFeatures(dim=m.zero_creator.dim)
    ln_out = None if type(m.ln_out).__name__ == "Identity" else from_torch(m.ln_out)
    mode = "example" if m.compact else m.mode
    return FeatureFactory(
        features=tuple(_convert_feature(f) for f in m.feat_creators),
        linear_out=from_torch(m.linear_out),
        ln_out=ln_out,
        mode=mode,
    )

