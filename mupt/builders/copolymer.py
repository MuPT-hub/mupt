"""High-level builders for random copolymer systems.

This module provides user-friendly functions for constructing polymer systems
from SMILES specifications, suitable for export to MD engines via the OpenFF
or MDAnalysis interfaces.
"""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'


import logging
from typing import Mapping, Iterable, Optional

import numpy as np
import networkx as nx

from ..mupr.primitives import Primitive
from ..mupr.topology import TopologicalStructure
from ..geometry.shapes import Ellipsoid
from ..geometry.coordinates.reference import origin
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.transforms.rigid import rigid_vector_coalignment
from ..interfaces.smiles import primitive_from_smiles
from ..interfaces.rdkit import suppress_rdkit_logs
from .random_walk import AngleConstrainedRandomWalk


logger = logging.getLogger(__name__)


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
        raise ValueError('Chain length must be at least 2 to accommodate head and tail units')
    
    yield head_name
    for mid_name in np.random.choice(
        list(mid_distrib.keys()),
        size=(chain_len - 2),
        p=list(mid_distrib.values()),
    ).astype(object):
        yield mid_name
    yield tail_name


def build_lexicon(
    rep_unit_smiles: dict[str, str],
    axis: int = 0,
    semiminor_fraction: float = 0.5,
) -> dict[str, Primitive]:
    """
    Build a lexicon of repeat unit Primitives from SMILES strings.
    
    Each repeat unit is oriented along the specified axis with an encompassing
    ellipsoidal shape for coarse-grained representation.
    
    Parameters
    ----------
    rep_unit_smiles : dict[str, str]
        Mapping from unit names to SMILES strings. SMILES should have
        atom map numbers [*:1] and [*:2] marking the head/tail connection sites.
    axis : int, default=0
        Axis along which to orient repeat units (0=X, 1=Y, 2=Z).
    semiminor_fraction : float, default=0.5
        Fraction of major radius for the minor axes of the ellipsoid.
        
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
                lambda prim: 'molAtomMapNumber' in prim.metadata,
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
            
            # Set encompassing ellipsoidal shape
            semiminor = semiminor_fraction * major_radius
            radii = np.full(3, semiminor)
            radii[axis] = major_radius
            unitprim.shape = Ellipsoid(radii)
            
            lexicon[unit_name] = unitprim
            logger.debug(f"Built repeat unit '{unit_name}': {len(unitprim.leaves)} atoms")
    
    return lexicon


def build_copolymer_system(
    rep_unit_smiles: dict[str, str],
    mid_distrib: dict[str, float],
    n_chains: int,
    chain_len_min: int,
    chain_len_max: int,
    head_name: str = 'head',
    tail_name: str = 'tail',
    bond_length: float = 1.5,
    angle_max_rad: float = np.pi / 4,
    exclusion_radius: float = 20.0,
    axis: int = 0,
    semiminor_fraction: float = 0.5,
    random_seed: Optional[int] = None,
    show_progress: bool = True,
) -> Primitive:
    """
    Build a random copolymer system with specified composition.
    
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
    semiminor_fraction : float, default=0.5
        Fraction of major radius for ellipsoid minor axes.
    random_seed : int, optional
        Random seed for reproducibility.
    show_progress : bool, default=True
        Whether to display a progress bar during construction.
        
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
    >>> univprim = build_copolymer_system(
    ...     rep_unit_smiles, mid_distrib,
    ...     n_chains=10, chain_len_min=5, chain_len_max=10,
    ... )
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Build lexicon of repeat units
    logger.info(f"Building lexicon with {len(rep_unit_smiles)} repeat unit types")
    lexicon = build_lexicon(rep_unit_smiles, axis=axis, semiminor_fraction=semiminor_fraction)
    
    # Validate that required units are present
    for required in [head_name, tail_name] + list(mid_distrib.keys()):
        if required not in lexicon:
            raise ValueError(f"Missing repeat unit '{required}' in rep_unit_smiles")
    
    # Build universe
    univprim = Primitive(label='universe')
    chain_lengths = np.random.randint(chain_len_min, chain_len_max + 1, size=n_chains)
    
    if show_progress:
        try:
            from rich.progress import Progress, BarColumn, TimeRemainingColumn
            progress_ctx = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
            )
        except ImportError:
            show_progress = False
    
    def _build_chains():
        for chain_idx, chain_len in enumerate(chain_lengths):
            # Build chain hierarchy
            molprim = Primitive(label=f'{chain_len}-mer_chain')
            
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
    
    if show_progress:
        with progress_ctx as progress:
            task = progress.add_task("Building chains", total=n_chains)
            for chain_idx, chain_len in _build_chains():
                progress.advance(task)
    else:
        for _ in _build_chains():
            pass
    
    total_atoms = len(univprim.leaves)
    total_residues = sum(len(chain.children) for chain in univprim.children)
    logger.info(
        f"Built system: {n_chains} chains, {total_residues} residues, {total_atoms} atoms"
    )
    
    return univprim

