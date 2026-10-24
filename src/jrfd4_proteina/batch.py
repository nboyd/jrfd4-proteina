"""Typed design inputs: everything the flow network, sampler and decoder read about one design problem.

`Batch` holds a single example (no batch axis); stack several with `jax.tree.map(jnp.stack, ...)` and
`jax.vmap`. Units follow the reference: conditioning coordinates in Å (NaN where unconditioned).
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jrfd4_proteina.constants import SEQUENCE_PROBABILITY_RESIDUES

Arr = Array | np.ndarray


class Batch(eqx.Module):
    # Token masks
    padding_mask: Bool[Arr, " N"]
    is_in_context_conditioning: Bool[Arr, " N"]
    supports_local_latent: Bool[Arr, " N"]
    coord_cond_mask: Bool[Arr, " N"]
    # Token identity
    token_type: Int[Arr, " N"]
    molecule_iid: Int[Arr, " N"]
    molecule_entity_id: Int[Arr, " N"]
    # Atom37 identity
    atom37_mask: Bool[Arr, "N 37"]
    charge: Int[Arr, "N 37"]
    element: Int[Arr, "N 37"]
    # Conditioning annotations
    cond_sequence: Int[Arr, " N"]
    sequence_probabilities: Float[Arr, "N R"]
    cond_coords: Float[Arr, "N 37 3"]
    cond_internal_coords: Float[Arr, "N 37 3"]
    coord_group: Int[Arr, "N 37"]
    rasa: Int[Arr, "N 37"]
    secondary_structure: Int[Arr, " N"]
    hbond: Int[Arr, "N 37"]
    pistack: Bool[Arr, "N 37"]
    pication: Int[Arr, "N 37"]
    saltbridge: Int[Arr, "N 37"]
    hydrophobic: Bool[Arr, "N 37"]
    metal_coord: Int[Arr, "N 37"]
    hotspot: Bool[Arr, "N 37"]
    cath: Int[Arr, "N 3"]
    index_group: Int[Arr, " N"]
    chain_centroid: Float[Arr, "N 3"]
    na_ss_state: Int[Arr, " N"]
    na_ss_state_mask: Bool[Arr, " N"]
    # Chirality
    chirality: Float[Arr, "N S3"]
    chiral_frame_atoms: Int[Arr, "N S K2"]
    chiral_slot_owner: Int[Arr, "N S"]
    # Pair features
    bond_mask: Bool[Arr, "N N"]
    bond_order: Int[Arr, "N N"]
    is_aromatic: Bool[Arr, "N N"]
    token_bond_path_length: Int[Arr, "N N"]
    canonical_polymer_ordering: Int[Arr, "N N"]
    na_ss_pair: Int[Arr, "N N"]
    na_ss_pair_mask: Bool[Arr, "N N"]

    @classmethod
    def from_reference(cls, arrays: Mapping[str, np.ndarray]) -> Batch:
        """Build from one example of the reference collated batch (key names as in `rfproteina`)."""
        renamed = {
            "cond_sequence": "condition_sequence_1_residue",
            "cond_coords": "condition_coordinate_1_atom",
            "cond_internal_coords": "condition_InternalCoordinate_1_atom",
            "coord_group": "condition_CoordinateGroup_1_atom",
            "rasa": "condition_rasa-binned_1_atom",
            "secondary_structure": "condition_secondary-structure_1_residue",
            "hbond": "condition_hbond_1_atom",
            "pistack": "condition_pistack_1_atom",
            "pication": "condition_pication_1_atom",
            "saltbridge": "condition_saltbridge_1_atom",
            "hydrophobic": "condition_hydrophobic_1_atom",
            "metal_coord": "condition_metal-coord_1_atom",
            "hotspot": "condition_hotspot_1_atom",
            "cath": "condition_cath-label_1_residue",
            "index_group": "condition_index-group_1_token",
            "chain_centroid": "condition_ChainCentroid_1_atom",
        }
        probabilities = np.stack(
            [arrays[f"condition_sequence-probability-{r}_1_residue"] for r in SEQUENCE_PROBABILITY_RESIDUES], -1
        )
        fields = {}
        for name in cls.__dataclass_fields__:
            if name == "sequence_probabilities":
                fields[name] = probabilities.astype(np.float32)
            else:
                fields[name] = np.asarray(arrays[renamed.get(name, name)])
        return cls(**fields)
