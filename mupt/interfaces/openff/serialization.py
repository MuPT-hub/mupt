"""Serialization utilities for OpenFF Molecule objects.

This module provides save/load functionality for OpenFF Molecule objects
with full metadata preservation, supporting both single molecules and
multi-molecule polymer systems.
"""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'


import json
import logging
from pathlib import Path
from typing import Union

from openff.toolkit import Molecule as OFFMolecule
from rdkit.Chem import SDWriter


logger = logging.getLogger(__name__)


def save_openff_system(
    molecules: Union[OFFMolecule, list[OFFMolecule]],
    filepath: Union[str, Path],
    also_save_sdf: bool = True,
) -> dict[str, Path]:
    """
    Save OpenFF Molecule(s) to JSON format with full metadata preservation.
    
    Handles both single molecules and multi-molecule systems. For polymer
    systems with multiple disconnected chains, pass a list of OFFMolecules.
    The JSON format preserves all OpenFF metadata including atom properties,
    stereochemistry, and hierarchy information.
    
    Parameters
    ----------
    molecules : OFFMolecule or list[OFFMolecule]
        Single molecule or list of molecules to save.
    filepath : str or Path
        Path to save JSON file. Extension will be added/replaced as needed.
    also_save_sdf : bool, default=True
        If True, also save an SDF file for RDKit compatibility. Note that
        SDF format does not preserve all OpenFF metadata.
        
    Returns
    -------
    dict[str, Path]
        Dictionary mapping format names to saved file paths:
        ``{'json': Path(...), 'sdf': Path(...)}``
        
    Examples
    --------
    Save a single molecule:
    
    >>> save_openff_system(offmol, "my_molecule")
    
    Save a multi-chain polymer system:
    
    >>> offmols = [OFFMolecule.from_rdkit(frag) for frag in fragments]
    >>> save_openff_system(offmols, "polymer_system")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize to list
    if isinstance(molecules, OFFMolecule):
        mol_list = [molecules]
    else:
        mol_list = list(molecules)
    
    json_path = filepath.with_suffix('.json')
    saved_files = {}
    
    # Save as JSON array
    # Use each molecule's to_json() to avoid bytes serialization issues,
    # then parse back to dicts so we can combine into one JSON file
    json_strings = [mol.to_json() for mol in mol_list]
    mol_dicts = [json.loads(s) for s in json_strings]
    
    with open(json_path, 'w') as f:
        json.dump(mol_dicts, f, indent=2)
    saved_files['json'] = json_path
    logger.info(f"Saved {len(mol_list)} molecule(s) to: {json_path}")
    
    # Optionally save SDF for RDKit workflows
    if also_save_sdf:
        sdf_path = filepath.with_suffix('.sdf')
        writer = SDWriter(str(sdf_path))
        
        for mol in mol_list:
            rdkit_mol = mol.to_rdkit()
            # Use V3000 format for large molecules
            if rdkit_mol.GetNumAtoms() > 999:
                writer.SetForceV3000(True)
            writer.write(rdkit_mol)
        writer.close()
        
        saved_files['sdf'] = sdf_path
        logger.info(f"Also saved SDF to: {sdf_path}")
    
    return saved_files


def load_openff_system(
    filepath: Union[str, Path],
    setup_hierarchy: bool = True,
) -> list[OFFMolecule]:
    """
    Load OpenFF Molecule(s) from JSON format.
    
    Restores molecules with all metadata preserved, including atom properties,
    stereochemistry, and conformer information. Optionally sets up hierarchy
    schemes for chain/residue iteration.
    
    Parameters
    ----------
    filepath : str or Path
        Path to JSON file created by ``save_openff_system``.
    setup_hierarchy : bool, default=True
        If True, add default hierarchy schemes to enable iteration over
        chains and residues via ``mol.chains`` and ``mol.residues``.
        
    Returns
    -------
    list[OFFMolecule]
        List of molecules with all metadata preserved.
        
    Examples
    --------
    Load and iterate over chains:
    
    >>> mols = load_openff_system("polymer_system.json")
    >>> for mol in mols:
    ...     for residue in mol.residues:
    ...         print(residue.residue_name)
    """
    filepath = Path(filepath)
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Handle both single-molecule (dict) and multi-molecule (list) formats
    if isinstance(data, dict):
        mol_dicts = [data]
    else:
        mol_dicts = data
    
    molecules = []
    for mol_dict in mol_dicts:
        # Convert dict back to JSON string for from_json
        mol = OFFMolecule.from_json(json.dumps(mol_dict))
        if setup_hierarchy:
            mol.add_default_hierarchy_schemes()
            mol.update_hierarchy_schemes()
        molecules.append(mol)
    
    logger.info(f"Loaded {len(molecules)} molecule(s) from: {filepath}")
    return molecules


def save_openff_molecule(
    molecule: OFFMolecule,
    filepath: Union[str, Path],
    **kwargs,
) -> dict[str, Path]:
    """
    Save a single OpenFF Molecule.
    
    Convenience wrapper around ``save_openff_system`` for single-molecule
    workflows.
    
    Parameters
    ----------
    molecule : OFFMolecule
        Single molecule to save.
    filepath : str or Path
        Path to save JSON file.
    **kwargs
        Additional arguments passed to ``save_openff_system``.
        
    Returns
    -------
    dict[str, Path]
        Dictionary of saved file paths.
        
    See Also
    --------
    save_openff_system : For multi-molecule systems.
    """
    return save_openff_system(molecule, filepath, **kwargs)


def load_openff_molecule(
    filepath: Union[str, Path],
    **kwargs,
) -> OFFMolecule:
    """
    Load a single OpenFF Molecule.
    
    Convenience wrapper around ``load_openff_system`` that expects exactly
    one molecule in the file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to JSON file.
    **kwargs
        Additional arguments passed to ``load_openff_system``.
        
    Returns
    -------
    OFFMolecule
        The loaded molecule.
        
    Raises
    ------
    ValueError
        If the file contains more than one molecule.
        
    See Also
    --------
    load_openff_system : For multi-molecule systems.
    """
    mols = load_openff_system(filepath, **kwargs)
    if len(mols) != 1:
        raise ValueError(
            f"Expected 1 molecule, found {len(mols)}. "
            "Use load_openff_system() for multi-molecule files."
        )
    return mols[0]
