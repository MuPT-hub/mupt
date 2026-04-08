"""
File to house various fixtures that are used by multiple tests.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

import pytest

import logging
from typing import Any, Generator, Mapping, Iterable, Optional

import numpy as np
import networkx as nx

from ..mupr.primitives import Primitive
from ..mupr.topology import TopologicalStructure
from ..geometry.coordinates.reference import origin
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.transforms.rigid import rigid_vector_coalignment
from ..interfaces.smiles import primitive_from_smiles
from ..interfaces.rdkit import suppress_rdkit_logs
from ..builders.random_walk import AngleConstrainedRandomWalk


logger = logging.getLogger(__name__)

# DEV:JRL The following functions are useful helpers to streamline the building
# of copolymer systems from SMILES. They were taken from the ellipsoidal_chain_placement.ipynb
# tutorial notebook authored by @timbernat


def sequence_repeat_units(
    chain_len: int,
    head_name: str,
    tail_name: str,
    mid_distrib: Mapping[str, float],
) -> Iterable[str]:
    """
    Generate a sequence of repeat unit names for a polymer chain.

    Parameters
    ----------
    chain_len : int
        Total length of the polymer chain (number of repeat units, including end groups).
    head_name : str
        Name of the head repeat unit.
    tail_name : str
        Name of the tail repeat unit.
    mid_distrib : Mapping[str, float]
        Distribution of mid repeat units (map from names to probabilities).
        Probabilities must sum to 1.0.

    Yields
    ------
    str
        Names of repeat units in sequence from head to tail.

    Raises
    ------
    ValueError
        If chain_len < 2 (need at least head and tail).
    """
    if chain_len < 2:
        raise ValueError(
            "Chain length must be at least 2 to accommodate head and tail units"
        )

    yield head_name
    for mid_name in np.random.choice(
        list(mid_distrib.keys()),
        size=(chain_len - 2),
        p=list(mid_distrib.values()),
    ).astype(object):
        yield mid_name
    yield tail_name


def build_SAAMR_lexicon(
    rep_unit_smiles: dict[str, str],
    axis: int = 0,
) -> dict[str, Primitive]:
    """
    Build a lexicon of repeat unit Primitives from SMILES strings.

    Each repeat unit is oriented along the specified axis. Atom positions
    are embedded via RDKit and aligned to the specified axis.

    Parameters
    ----------
    rep_unit_smiles : dict[str, str]
        Mapping from unit names to SMILES strings. SMILES should have
        atom map numbers [*:1] and [*:2] marking the head/tail connection sites.
    axis : int, default=0
        Axis along which to orient repeat units (0=X, 1=Y, 2=Z).

    Returns
    -------
    dict[str, Primitive]
        Lexicon mapping unit names to oriented Primitive objects.
    """
    lexicon: dict[str, Primitive] = {}

    with suppress_rdkit_logs():
        for unit_name, smiles in rep_unit_smiles.items():
            unitprim = primitive_from_smiles(
                smiles,
                ensure_explicit_Hs=True,
                embed_positions=True,
                label=unit_name,
            )

            # Find edge atoms marked with atom map numbers
            head_atom, tail_atom = unitprim.search_hierarchy_by(
                lambda prim: "molAtomMapNumber" in prim.metadata,
                min_count=2,
            )
            head_pos = head_atom.shape.centroid
            tail_pos = tail_atom.shape.centroid

            # Orient along specified axis
            major_radius = np.linalg.norm(tail_pos - head_pos) / 2.0
            axis_vec = np.zeros(3, dtype=float)
            axis_vec[axis] = major_radius

            axis_alignment = rigid_vector_coalignment(
                vector1_start=head_pos,
                vector1_end=tail_pos,
                vector2_start=origin(3),
                vector2_end=axis_vec,
                t1=0.5,
                t2=0.0,
            )
            unitprim.rigidly_transform(axis_alignment)

            lexicon[unit_name] = unitprim
            logger.debug(
                f"Built repeat unit '{unit_name}': {len(unitprim.leaves)} atoms"
            )

    return lexicon


def build_SAAMR_polymer_system(
    rep_unit_smiles: dict[str, str],
    mid_distrib: dict[str, float],
    n_chains: int,
    chain_len_min: int,
    chain_len_max: int,
    head_name: str = "head",
    tail_name: str = "tail",
    bond_length: float = 1.5,
    angle_max_rad: float = np.pi / 4,
    exclusion_radius: float = 20.0,
    axis: int = 0,
    random_seed: Optional[int] = None,
) -> Primitive:
    """
    Build a (random co)polymer system with specified composition.

    Constructs a universe Primitive containing multiple polymer chains with
    random lengths and monomer sequences according to the specified distribution.
    Chains are placed using an angle-constrained random walk.

    The resulting Primitive hierarchy is structured as:
    universe -> chains -> residues (repeat units) -> atoms

    Parameters
    ----------
    rep_unit_smiles : dict[str, str]
        Mapping from unit names to SMILES strings. Must include entries for
        head_name and tail_name. SMILES should have atom map numbers [*:1] and
        [*:2] marking head/tail connection sites.
    mid_distrib : dict[str, float]
        Distribution of middle repeat units (probabilities must sum to 1.0).
    n_chains : int
        Number of polymer chains to generate.
    chain_len_min : int
        Minimum chain length (number of repeat units including end groups).
    chain_len_max : int
        Maximum chain length (inclusive).
    head_name : str, default='head'
        Name of the head repeat unit in rep_unit_smiles.
    tail_name : str, default='tail'
        Name of the tail repeat unit in rep_unit_smiles.
    bond_length : float, default=1.5
        Distance between repeat unit centers along the chain.
    angle_max_rad : float, default=π/4
        Maximum bend angle (radians) between consecutive bonds.
    exclusion_radius : float, default=20.0
        Radius from origin at which to start placing chains.
    axis : int, default=0
        Axis along which to orient repeat units (0=X, 1=Y, 2=Z).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Primitive
        Universe Primitive containing the complete polymer system.

    Examples
    --------
    Build a PSU/PES copolymer system:

    >>> rep_unit_smiles = {
    ...     'head': '[H]-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
    ...     'bisphenol_S': '*-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
    ...     'bisphenol_A': '*-[O:1]c1ccc(cc1)C(-C)(-C)c1cc[c:2](cc1)-*',
    ...     'tail': '*-[O:1]c1ccc(cc1)S(=O)(=O)c1ccc(cc1)[O:2]-[H]',
    ... }
    >>> mid_distrib = {'bisphenol_S': 0.4, 'bisphenol_A': 0.6}
    >>> univprim = build_SAAMR_polymer_system(
    ...     rep_unit_smiles, mid_distrib,
    ...     n_chains=10, chain_len_min=5, chain_len_max=10,
    ... )
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Build lexicon of repeat units
    logger.info(f"Building lexicon with {len(rep_unit_smiles)} repeat unit types")
    lexicon = build_SAAMR_lexicon(rep_unit_smiles, axis=axis)

    # Validate that required units are present
    for required in [head_name, tail_name] + list(mid_distrib.keys()):
        if required not in lexicon:
            raise ValueError(f"Missing repeat unit '{required}' in rep_unit_smiles")

    # Build universe
    univprim = Primitive(label="universe")
    chain_lengths = np.random.randint(chain_len_min, chain_len_max + 1, size=n_chains)

    def _build_chains() -> Generator[tuple[int, Any], Any, None]:
        for chain_idx, chain_len in enumerate(chain_lengths):
            # Build chain hierarchy
            molprim = Primitive(label=f"{chain_len}-mer_chain")

            for unit_name in sequence_repeat_units(
                chain_len,
                head_name=head_name,
                tail_name=tail_name,
                mid_distrib=mid_distrib,
            ):
                rep_unit_prim = lexicon[unit_name].copy()
                molprim.attach_child(rep_unit_prim)

            # Set path graph topology
            molprim.set_topology(
                nx.path_graph(
                    molprim.children_by_handle.keys(),
                    create_using=TopologicalStructure,
                ),
                max_registration_iter=100,
            )

            # Place beads by random walk
            direction = random_unit_vector()
            # DEV-JRL: This is where we should think about inserting the DPD Builder!
            builder = AngleConstrainedRandomWalk(
                bond_length=bond_length,
                angle_max_rad=angle_max_rad,
                initial_point=exclusion_radius * direction,
                initial_direction=direction,
            )

            for handle, placement in builder.generate_placements(molprim):
                molprim.children_by_handle[handle].rigidly_transform(placement)

            # Attach chain to universe
            univprim.attach_child(molprim)

            yield chain_idx, chain_len

    # Consume the generator to actually build all chains
    for _ in _build_chains():
        pass

    total_atoms = len(univprim.leaves)
    total_residues = sum(len(chain.children) for chain in univprim.children)
    logger.info(
        f"Built system: {n_chains} chains, {total_residues} residues, {total_atoms} atoms"
    )

    return univprim


@pytest.fixture
def polyethylene_smiles() -> dict[str, str]:
    """SMILES definitions for polyethylene"""
    return {
        "head": "[H:1]-[CH2:2]-*",
        "ethane": "*-[CH2:1][CH2:2]-*",
        "tail": "*-[CH2:1]-[H:2]",
    }


# resname_maps are currently used to assign PDB compliant 3-char codes
# to repeat units
@pytest.fixture
def polyethylene_resname_map() -> dict[str, str]:
    """Residue name mapping for polyethylene systems."""
    return {"head": "HEA", "ethane": "EAN", "tail": "TYL"}


@pytest.fixture
def polyethersulfone_smiles() -> dict[str, str]:
    """SMILES definitions for BPA/BPS copolymer, aka polyethersulfone (PES)"""
    return {
        "head": "[H]-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*",
        "bisphenol_S": "*-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*",
        "bisphenol_A": "*-[O:1]c1ccc(cc1)C(-C)(-C)c1cc[c:2](cc1)-*",
        "tail": "*-[O:1]c1ccc(cc1)S(=O)(=O)c1ccc(cc1)[O:2]-[H]",
    }


@pytest.fixture
def PES_resname_map() -> dict[str, str]:
    """Residue name mapping for BPA/BPS copolymer systems."""
    return {"head": "HED", "bisphenol_S": "BPS", "bisphenol_A": "BPA", "tail": "TAL"}


@pytest.fixture
def single_polyethylene_2mer(
    polyethylene_smiles: dict[str, str],
) -> Primitive:
    """
    Fixture providing a Primitive containing a single molecule of
    polyethylene composed of 2 repeat units of ethane.
    Primitive is intended to be SAAMR-compliant.
    [Universe -> Molecule -> Repeat-Units -> Atoms]

    * should have:
    - 1 chain
    - 2 repeat units
    - 8 atoms
    - 6 intra-residue bonds
    - 1 inter-residue bond
    """
    return build_SAAMR_polymer_system(
        polyethylene_smiles,
        mid_distrib={"ethane": 1.0},
        n_chains=1,
        chain_len_min=2,
        chain_len_max=2,
        random_seed=42,
    )


@pytest.fixture
def single_polyethylene_3mer(
    polyethylene_smiles: dict[str, str],
) -> Primitive:
    """
    Fixture providing a Primitive containing a single molecule of
    polyethylene composed of 3 repeat units of ethane.
    Primitive is intended to be SAAMR-compliant.
    [Universe -> Molecule -> Repeat-Units -> Atoms]

    * should have:
    - 1 chain
    - 3 repeat units
    - 14 atoms
    - 11 intra-residue bonds
    - 2 inter-residue bonds
    """
    return build_SAAMR_polymer_system(
        polyethylene_smiles,
        mid_distrib={"ethane": 1.0},
        n_chains=1,
        chain_len_min=3,
        chain_len_max=3,
        random_seed=42,
    )


@pytest.fixture
def multi_polyethylene_system(
    polyethylene_smiles: dict[str, str],
) -> Primitive:
    """
    Fixture providing a multi-chain polyethylene system with varying chain lengths.
    Useful for testing system-level operations.

    * should have:
    - 10 chains
    - 5-10 repeat units per chain
    - Variable total atoms/bonds depending on random chain lengths
    """
    return build_SAAMR_polymer_system(
        polyethylene_smiles,
        mid_distrib={"ethane": 1.0},
        n_chains=10,
        chain_len_min=5,
        chain_len_max=10,
        random_seed=42,
    )


@pytest.fixture
def PES_copolymer(polyethersulfone_smiles: dict[str, str]) -> Primitive:
    """
    Fixture providing a default BPA/BPS copolymer system Primitive.
    Primitive is intended to be SAAMR-compliant.
    [Universe -> Molecule -> Repeat-Units -> Atoms]

    Default configuration: 5 chains, 5-10 repeat units per chain, 40% BPS / 60% BPA
    """
    return build_SAAMR_polymer_system(
        polyethersulfone_smiles,
        mid_distrib={"bisphenol_S": 0.4, "bisphenol_A": 0.6},
        n_chains=5,
        chain_len_min=5,
        chain_len_max=10,
        random_seed=42,
    )


@pytest.fixture
def helium_resname_map() -> dict[str, str]:
    """Residue name mapping for Helium SAAMR system."""
    return {"He_unit": "HEL"}


@pytest.fixture
def single_helium_atom_saamr() -> Primitive:
    """
    Fixture providing the simplest possible SAAMR-compliant system: a single Helium atom.

    This serves as a base case for testing with minimal complexity.
    Hierarchy: [Universe -> Molecule -> Repeat-Unit -> Atom]

    * should have:
    - 1 molecule
    - 1 repeat unit
    - 1 atom (He) at position [0, 0, 0]
    """
    from ..geometry.shapes import Sphere
    from periodictable import elements

    # Create the atom-level primitive (Helium)
    atom_prim = Primitive(label="He")
    atom_prim.element = elements.He  # Set element to make it an atom
    atom_prim.shape = Sphere(0.49)  # 0.49 Angstrom radius at origin

    # Create the repeat unit and attach the atom
    repeat_unit_prim = Primitive(label="He_unit")
    repeat_unit_prim.attach_child(atom_prim)

    # Create the molecule and attach the repeat unit
    molecule_prim = Primitive(label="He_molecule")
    molecule_prim.attach_child(repeat_unit_prim)

    # Create the universe and attach the molecule
    universe_prim = Primitive(label="universe")
    universe_prim.attach_child(molecule_prim)

    return universe_prim
