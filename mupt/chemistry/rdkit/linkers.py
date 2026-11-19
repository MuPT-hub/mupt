"""Perception and manipulation of wild-type "linker" atoms in an RDKit molecule"""

from typing import Generator

from rdkit import Chem
from rdkit.Chem import Atom, Mol


# DEVNOTE: unclear whether X (total connections) or D (explicit connections)
# is the right choice for this query or if there's ever a case where the two
# would not produce identical results; both seem to handle higher-order bonds
# correctly (i.e. treat double bond as "one" connection)

# neutronium-excluding linker query; requires that
# the linker be attached to a non-linker atom
LINKER_QUERY = "[!#0]~[#0X1]"
## Exactly 1 of anything (including possibly another
## wild-type atom) attached to atomic number 0 atom
# LINKER_QUERY : str = '*~[#0X1]'
LINKER_QUERY_MOL: Mol = Chem.MolFromSmarts(LINKER_QUERY)


def is_linker(rdatom: Atom) -> bool:
    """Indicate whether an atom is a linker (intermonomer "*" type atom)"""
    return rdatom.GetAtomicNum() == 0


def not_linker(rdatom: Atom) -> bool:
    """Indicate whether an atom is NOT a linker, i.e. is a "real" atom"""
    # return rdatom.GetAtomicNum() != 0
    return not is_linker(rdatom)


is_real_atom = not_linker


def num_linkers(rdmol: Mol) -> int:
    """Count how many wild-type inter-molecule linker atoms are in a Mol"""
    return sum(is_linker(atom) for atom in rdmol.GetAtoms())


def anchor_and_linker_idxs(rdmol: Mol) -> Generator[tuple[int, int], None, None]:
    """Get the anchor and linker indices of all ports found in an RDKit Mol"""
    # DON'T de-duplify (i.e. uniquify) indices of substruct
    # matches (fails to catch both ports on a neutronium)
    for anchor_idx, linker_idx in rdmol.GetSubstructMatches(
        LINKER_QUERY_MOL, uniquify=False
    ):
        yield anchor_idx, linker_idx  # unpacked purely for self-documentation


def real_and_linker_atom_idxs(rdmol: Mol) -> tuple[list[int], list[int]]:
    """
    Partition the atom indices of a Mol by whether
    or not their corresponding atom is a linker atom

    Returns a pair of lists, the first containing the indices of non-linker
    (i.e. "real") atoms and the second containing indices of linkers
    """
    idxs_partitioned_by_linker = ([], [])
    for atom in rdmol.GetAtoms():
        idxs_partitioned_by_linker[is_linker(atom)].append(atom.GetIdx())

    return idxs_partitioned_by_linker


def renumber_linkers_as_last(rdmol: Mol) -> Mol:  # TODO: make optionally in-place
    """
    Returns a copy of a Mol whose atom indices are renumbered such that:

    * all #L linker atoms are assigned the last L indices
      (i.e. those which occur after all real atoms in order)
    * all non-linker (i.e. "real") atom are numbered
      in the order they appear in the original Mol
    """
    real_atom_idxs, linker_idxs = real_and_linker_atom_idxs(rdmol)

    return Chem.RenumberAtoms(
        rdmol, real_atom_idxs + linker_idxs
    )  # place linker indices at end of list


def embed_with_linker_placeholders(
    rdmol: Mol,
    random_seed: int = -1,
    placeholder_atomic_num: int = 6,
) -> int:
    """
    Embed a 3D conformer into a Mol in-place, returning its ID (negative on failure)

    Linker atoms are temporarily swapped for hydrogen-free placeholder atoms (carbon by
    default) while embedding; this sidesteps RDKit distance-geometry failures on wild-type
    atoms (e.g. two linkers 4 bonds apart) and places linkers at realistic bond lengths.
    Chiral tags are respected by the embedding
    """
    from rdkit.Chem.rdDistGeom import EmbedMolecule  # local import keeps module light

    placeholder_mol = Chem.RWMol(rdmol)
    for atom in placeholder_mol.GetAtoms():
        if is_linker(atom):
            atom.SetAtomicNum(placeholder_atomic_num)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    placeholder_mol.UpdatePropertyCache(strict=False)

    conformer_id = EmbedMolecule(placeholder_mol, randomSeed=random_seed)
    if conformer_id < 0:
        return conformer_id

    # atom indices are unchanged, so the conformer transfers directly
    return rdmol.AddConformer(
        Chem.Conformer(placeholder_mol.GetConformer(conformer_id)),
        assignId=True,
    )
