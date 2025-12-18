"""Exporters for converting MuPT Primitives to OpenFF Molecules.

This module provides high-level functions for converting MuPT Primitive
hierarchies to OpenFF Molecule and Topology objects, suitable for
parameterization with OpenFF force fields.
"""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'


import logging
from typing import Optional

from openff.toolkit import Molecule as OFFMolecule
from openff.toolkit import Topology as OFFTopology
from rdkit import Chem

from ..rdkit.exporters import primitive_to_rdkit_hierarchical
from ...mupr.primitives import Primitive


logger = logging.getLogger(__name__)


def _is_AA_export_compliant(prim: Primitive) -> bool:
    """
    Check whether a Primitive hierarchy is organized
    as universe -> chain -> residue -> atom.
    
    This structure is required for proper export to OpenFF with
    chain/residue metadata preservation.
    """
    return all(
        leaf.is_atom and (leaf.depth == 3)
        for leaf in prim.leaves
    )


def primitive_to_openff_molecules(
    primitive: Primitive,
    resname_map: Optional[dict[str, str]] = None,
    allow_undefined_stereo: bool = True,
) -> list[OFFMolecule]:
    """
    Convert a MuPT Primitive hierarchy to a list of OpenFF Molecules.
    
    This is the primary export function for converting MuPT-built polymer
    systems to OpenFF format for force field parameterization. The function:
    
    1. Exports the Primitive to RDKit with full hierarchy metadata
       (chain IDs, residue names/numbers via AtomPDBResidueInfo)
    2. Splits disconnected fragments into separate molecules
    3. Converts each fragment to an OpenFF Molecule with hierarchy schemes
    
    The resulting molecules preserve residue/chain metadata and can be
    iterated via ``mol.chains`` and ``mol.residues``.
    
    Parameters
    ----------
    primitive : Primitive
        The root primitive (universe level) containing the molecular system.
        Must be organized as: universe -> chains -> residues -> atoms.
    resname_map : dict[str, str], optional
        Mapping from MuPT residue labels to 3-character PDB residue names.
        For example: ``{'head': 'HEA', 'middle': 'MID', 'tail': 'TAL'}``.
        If a label is not in the map, the first 3 characters are used.
    allow_undefined_stereo : bool, default=True
        If True, allow molecules with undefined stereochemistry. Set to
        False to raise errors for ambiguous stereocenters.
        
    Returns
    -------
    list[OFFMolecule]
        List of OpenFF Molecules, one per disconnected fragment (chain).
        Each molecule has hierarchy schemes set up for residue iteration.
        
    Examples
    --------
    Convert a polymer system and iterate residues:
    
    >>> from mupt.interfaces.openff import primitive_to_openff_molecules
    >>> offmols = primitive_to_openff_molecules(univprim, resname_map)
    >>> for mol in offmols:
    ...     print(f"Chain with {mol.n_atoms} atoms")
    ...     for res in mol.residues:
    ...         print(f"  Residue: {res.residue_name}")
    
    See Also
    --------
    primitive_to_openff_topology : For creating an OpenFF Topology directly.
    save_openff_system : For saving the molecules to disk.
    
    Raises
    ------
    ValueError
        If the Primitive hierarchy is not organized as
        universe -> chains -> residues -> atoms.
    """
    # Validate hierarchy structure
    if not _is_AA_export_compliant(primitive):
        raise ValueError(
            "Primitive must be organized as: universe -> chains -> residues -> atoms. "
            "All leaf nodes must be atoms at depth 3."
        )
    
    # Step 1: Export to RDKit with full hierarchy metadata
    rdkit_mol = primitive_to_rdkit_hierarchical(primitive, resname_map)
    logger.debug(f"Exported Primitive to RDKit: {rdkit_mol.GetNumAtoms()} atoms")
    
    # Step 2: Split into disconnected fragments
    # GetMolFrags preserves AtomPDBResidueInfo on each fragment
    fragments = Chem.GetMolFrags(rdkit_mol, asMols=True)
    logger.info(f"System has {len(fragments)} disconnected molecule(s)")
    
    # Step 3: Convert each fragment to OpenFF Molecule
    offmols = []
    for i, frag in enumerate(fragments):
        offmol = OFFMolecule.from_rdkit(
            frag,
            hydrogens_are_explicit=True,
            allow_undefined_stereo=allow_undefined_stereo,
        )
        
        # Set up hierarchy schemes for chain/residue iteration
        offmol.add_default_hierarchy_schemes()
        offmol.update_hierarchy_schemes()
        
        n_residues = len(list(offmol.residues))
        logger.debug(f"  Fragment {i+1}: {offmol.n_atoms} atoms, {n_residues} residues")
        
        offmols.append(offmol)
    
    return offmols


def primitive_to_openff_topology(
    primitive: Primitive,
    resname_map: Optional[dict[str, str]] = None,
    allow_undefined_stereo: bool = True,
) -> OFFTopology:
    """
    Convert a MuPT Primitive hierarchy to an OpenFF Topology.
    
    Creates a combined topology from all molecules in the system, suitable
    for use with OpenFF Interchange for MD engine export.
    
    Parameters
    ----------
    primitive : Primitive
        The root primitive (universe level) containing the molecular system.
    resname_map : dict[str, str], optional
        Mapping from MuPT residue labels to 3-character PDB residue names.
    allow_undefined_stereo : bool, default=True
        If True, allow molecules with undefined stereochemistry.
        
    Returns
    -------
    OFFTopology
        OpenFF Topology containing all molecules in the system.
        
    Examples
    --------
    Create topology for parameterization:
    
    >>> from mupt.interfaces.openff import primitive_to_openff_topology
    >>> topology = primitive_to_openff_topology(univprim, resname_map)
    >>> print(f"Topology: {topology.n_atoms} atoms, {topology.n_molecules} molecules")
    
    See Also
    --------
    primitive_to_openff_molecules : For a list of individual molecules.
    """
    offmols = primitive_to_openff_molecules(
        primitive,
        resname_map=resname_map,
        allow_undefined_stereo=allow_undefined_stereo,
    )
    
    topology = OFFTopology.from_molecules(offmols)
    logger.info(
        f"Created OpenFF Topology: {topology.n_atoms} atoms, "
        f"{topology.n_molecules} molecules"
    )
    
    return topology
