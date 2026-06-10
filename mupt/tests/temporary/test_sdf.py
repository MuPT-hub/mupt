"""Tests for temporary MuPT SDF interoperability export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from rdkit.Chem.rdmolfiles import SDMolSupplier

import mupt.interfaces.rdkit.exporters as rdkit_exporters
from mupt.interfaces.rdkit import write_primitive_to_mupt_sdf
from mupt.temporary.sdf import MUPT_SDF_ATOM_PROPS, write_primitive_to_sdf


def _load_sdf(path):
    return [mol for mol in SDMolSupplier(str(path), removeHs=False, sanitize=False) if mol is not None]


def _atoms_by_mupt_residue(mol):
    """Group SDF-loaded atoms by their original MuPT residue index."""
    atoms_by_residue = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("mupt_residue_index"):
            residue_index = int(atom.GetProp("mupt_residue_index"))
            atoms_by_residue.setdefault(residue_index, []).append(atom)
    return atoms_by_residue


def _sdf_boundary_bonds(mol):
    """Find SDF-loaded bonds crossing the artificial A:2 to B:1 boundary."""
    boundary_bonds = []
    for bond in mol.GetBonds():
        atoms = (bond.GetBeginAtom(), bond.GetEndAtom())
        if not all(
            atom.HasProp("mupt_residue_index") and atom.HasProp("chain_id")
            for atom in atoms
        ):
            continue
        residue_indices = {int(atom.GetProp("mupt_residue_index")) for atom in atoms}
        chain_ids = {atom.GetProp("chain_id") for atom in atoms}
        if residue_indices == {2, 3} and chain_ids == {"A", "B"}:
            boundary_bonds.append(bond)
    return boundary_bonds


def test_write_primitive_to_sdf_writes_one_record_per_segment(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    sdf_path = tmp_path / "chains.sdf"

    records = write_primitive_to_sdf(
        multi_polyethylene_system,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    mols = _load_sdf(sdf_path)

    assert records == len(multi_polyethylene_system.children)
    assert len(mols) == records


def test_write_primitive_to_sdf_preserves_openff_atom_metadata(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    sdf_path = tmp_path / "chain.sdf"

    write_primitive_to_sdf(
        single_polyethylene_2mer,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    mol = _load_sdf(sdf_path)[0]
    first_atom = mol.GetAtomWithIdx(0)

    for prop_name in MUPT_SDF_ATOM_PROPS:
        assert mol.HasProp(f"atom.prop.{prop_name}")
    assert first_atom.GetProp("residue_name") in set(polyethylene_resname_map.values())
    assert first_atom.HasProp("chain_id")


def test_rdkit_interface_reexports_mupt_sdf_writer(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    sdf_path = tmp_path / "chain.sdf"

    records = write_primitive_to_mupt_sdf(
        single_polyethylene_2mer,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )

    assert records == 1
    assert len(_load_sdf(sdf_path)) == 1


def test_write_primitive_to_sdf_roundtrip_wraps_pdb_residue_atom_props(
    tmp_path,
    single_polyethylene_3mer,
    polyethylene_resname_map,
    monkeypatch,
):
    """SDF round-trip preserves wrapped PDB-compatible residue atom props."""
    monkeypatch.setattr(rdkit_exporters, "PDB_MAX_RESIDUE_NUMBER", 2)
    sdf_path = tmp_path / "wrapped-chain.sdf"

    records = write_primitive_to_sdf(
        single_polyethylene_3mer,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    mols = _load_sdf(sdf_path)

    assert records == 1
    assert len(mols) == 1
    atoms_by_residue = _atoms_by_mupt_residue(mols[0])

    assert set(atoms_by_residue) == {1, 2, 3}
    for mupt_residue_index, expected_surrogate_id in {
        1: ("A", 1),
        2: ("A", 2),
        3: ("B", 1),
    }.items():
        for atom in atoms_by_residue[mupt_residue_index]:
            assert atom.HasProp("mupt_residue_index")
            assert (atom.GetProp("chain_id"), int(atom.GetProp("residue_id"))) == expected_surrogate_id


def test_write_primitive_to_sdf_roundtrip_preserves_wrapped_boundary_bond(
    tmp_path,
    single_polyethylene_3mer,
    polyethylene_resname_map,
    monkeypatch,
):
    """SDF round-trip keeps the bond across the artificial A-to-B boundary."""
    monkeypatch.setattr(rdkit_exporters, "PDB_MAX_RESIDUE_NUMBER", 2)
    sdf_path = tmp_path / "wrapped-boundary.sdf"

    write_primitive_to_sdf(
        single_polyethylene_3mer,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    mol = _load_sdf(sdf_path)[0]

    assert len(_sdf_boundary_bonds(mol)) == 1
