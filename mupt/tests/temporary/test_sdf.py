"""Tests for temporary MuPT SDF interoperability export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

import subprocess
import sys

import numpy as np
import pytest
from anytree import PreOrderIter
from rdkit.Chem.rdmolfiles import SDMolSupplier, SDWriter

import mupt.interfaces.rdkit.exporters as rdkit_exporters
from mupt.interfaces.rdkit import write_primitive_to_mupt_sdf
from mupt.roles import PrimitiveRole
from mupt.temporary.sdf import (
    MUPT_SDF_ATOM_PROPS,
    primitive_from_mupt_sdf,
    prepare_mupt_sdf_atom_props,
    write_primitive_to_sdf,
)


def _load_sdf(path):
    return [mol for mol in SDMolSupplier(str(path), removeHs=False, sanitize=False) if mol is not None]


def _mupt_sdf_path(path):
    if str(path).endswith(".mupt.sdf"):
        return path
    if path.suffix == ".sdf":
        return path.with_suffix(".mupt.sdf")
    return path.with_name(f"{path.name}.mupt.sdf")


def _total_internal_bonds(primitive):
    return sum(node.num_internal_connections for node in PreOrderIter(primitive))


def _atom_mupt_props(mol):
    props = []
    for atom in mol.GetAtoms():
        props.append(
            tuple(
                atom.GetProp(prop_name) if atom.HasProp(prop_name) else None
                for prop_name in MUPT_SDF_ATOM_PROPS
            )
        )
    return props


def _atom_positions(mol):
    conformer = mol.GetConformer(0)
    return np.array(
        [conformer.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()],
        dtype=float,
    )


def _record_props(mol, prop_names):
    return {prop_name: mol.GetProp(prop_name) for prop_name in prop_names}


def _write_mol(path, mol):
    writer = SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


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


def test_sdf_canonical_import_path_works_first_in_clean_interpreter():
    """Importing mupt.temporary.sdf first must not circular-import RDKit helpers."""
    code = """
from mupt.temporary.sdf import write_primitive_to_sdf
from mupt.interfaces.rdkit import write_primitive_to_sdf as compat_write
assert callable(write_primitive_to_sdf)
assert callable(compat_write)
"""

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


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
    mols = _load_sdf(_mupt_sdf_path(sdf_path))

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
    mol = _load_sdf(_mupt_sdf_path(sdf_path))[0]
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
    assert len(_load_sdf(_mupt_sdf_path(sdf_path))) == 1


def test_write_primitive_to_sdf_normalizes_mupt_sdf_paths(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    """Temporary SDF export always writes the MuPT-specific SDF suffix."""
    for path in [
        tmp_path / "no-suffix",
        tmp_path / "sdf-suffix.sdf",
        tmp_path / "mupt-suffix.mupt.sdf",
    ]:
        records = write_primitive_to_sdf(
            single_polyethylene_2mer,
            path,
            resname_map=polyethylene_resname_map,
        )

        assert records == 1
        assert _mupt_sdf_path(path).exists()


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
    mols = _load_sdf(_mupt_sdf_path(sdf_path))

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
    mol = _load_sdf(_mupt_sdf_path(sdf_path))[0]

    assert len(_sdf_boundary_bonds(mol)) == 1


def test_primitive_from_mupt_sdf_roundtrips_exportable_hierarchy(
    tmp_path,
    single_polyethylene_3mer,
    polyethylene_resname_map,
):
    """Imported temporary SDF hierarchies can be exported again without data loss."""
    first_path = tmp_path / "first.mupt.sdf"
    second_path = tmp_path / "second.mupt.sdf"
    single_polyethylene_3mer.metadata["root_tag"] = "root-value"
    single_polyethylene_3mer.children[0].metadata["segment_tag"] = "segment-value"

    first_records = write_primitive_to_sdf(
        single_polyethylene_3mer,
        first_path,
        resname_map=polyethylene_resname_map,
    )
    rebuilt = primitive_from_mupt_sdf(first_path)
    second_records = write_primitive_to_sdf(
        rebuilt,
        second_path,
        resname_map=polyethylene_resname_map,
    )
    first_mols = _load_sdf(first_path)
    second_mols = _load_sdf(second_path)

    assert first_records == second_records == 1
    assert [node.role for node in PreOrderIter(rebuilt)] == [
        PrimitiveRole.UNIVERSE,
        PrimitiveRole.SEGMENT,
        PrimitiveRole.RESIDUE,
        *([PrimitiveRole.PARTICLE] * 4),
        PrimitiveRole.RESIDUE,
        *([PrimitiveRole.PARTICLE] * 6),
        PrimitiveRole.RESIDUE,
        *([PrimitiveRole.PARTICLE] * 4),
    ]
    assert rebuilt.num_atoms == single_polyethylene_3mer.num_atoms
    assert _total_internal_bonds(rebuilt) == _total_internal_bonds(single_polyethylene_3mer)
    assert [mol.GetNumAtoms() for mol in first_mols] == [mol.GetNumAtoms() for mol in second_mols]
    assert [mol.GetNumBonds() for mol in first_mols] == [mol.GetNumBonds() for mol in second_mols]
    assert [_atom_mupt_props(mol) for mol in first_mols] == [
        _atom_mupt_props(mol) for mol in second_mols
    ]
    assert [_record_props(mol, ["root_tag", "segment_tag"]) for mol in first_mols] == [
        _record_props(mol, ["root_tag", "segment_tag"]) for mol in second_mols
    ]
    assert rebuilt.children[0].metadata["root_tag"] == "root-value"
    assert rebuilt.children[0].metadata["segment_tag"] == "segment-value"
    for first_mol, second_mol in zip(first_mols, second_mols):
        np.testing.assert_allclose(_atom_positions(first_mol), _atom_positions(second_mol))


def test_primitive_from_mupt_sdf_roundtrips_multi_record_system(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    """The importer rebuilds one SEGMENT hierarchy for each SDF record."""
    first_path = tmp_path / "multi-first.mupt.sdf"
    second_path = tmp_path / "multi-second.mupt.sdf"

    first_records = write_primitive_to_sdf(
        multi_polyethylene_system,
        first_path,
        resname_map=polyethylene_resname_map,
    )
    rebuilt = primitive_from_mupt_sdf(first_path)
    second_records = write_primitive_to_sdf(
        rebuilt,
        second_path,
        resname_map=polyethylene_resname_map,
    )
    first_mols = _load_sdf(first_path)
    second_mols = _load_sdf(second_path)

    assert first_records == second_records == len(multi_polyethylene_system.children)
    assert len(rebuilt.children) == len(multi_polyethylene_system.children)
    assert [mol.GetNumAtoms() for mol in first_mols] == [mol.GetNumAtoms() for mol in second_mols]
    assert [mol.GetNumBonds() for mol in first_mols] == [mol.GetNumBonds() for mol in second_mols]
    assert [_atom_mupt_props(mol) for mol in first_mols] == [
        _atom_mupt_props(mol) for mol in second_mols
    ]


def test_primitive_from_mupt_sdf_rejects_invalid_records(tmp_path):
    """Invalid SDF records should fail instead of returning a partial hierarchy."""
    sdf_path = tmp_path / "invalid.mupt.sdf"
    sdf_path.write_text("not an sdf record\n$$$$\n")

    with pytest.raises(ValueError, match="Could not parse MuPT SDF record"):
        primitive_from_mupt_sdf(sdf_path)


def test_primitive_from_mupt_sdf_rejects_atoms_missing_mupt_props(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    """Malformed MuPT records must not silently drop non-linker atoms."""
    valid_path = tmp_path / "valid.mupt.sdf"
    malformed_path = tmp_path / "malformed.mupt.sdf"
    write_primitive_to_sdf(
        single_polyethylene_2mer,
        valid_path,
        resname_map=polyethylene_resname_map,
    )
    mol = _load_sdf(valid_path)[0]
    mol.GetAtomWithIdx(0).ClearProp("mupt_particle_label")
    prepare_mupt_sdf_atom_props(mol)
    _write_mol(malformed_path, mol)

    with pytest.raises(ValueError, match="lacks required MuPT particle props"):
        primitive_from_mupt_sdf(malformed_path)


def test_primitive_from_mupt_sdf_rejects_inconsistent_segment_props(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    """All particle atoms in one record must agree on SEGMENT identity."""
    valid_path = tmp_path / "valid.mupt.sdf"
    malformed_path = tmp_path / "bad-segment.mupt.sdf"
    write_primitive_to_sdf(
        single_polyethylene_2mer,
        valid_path,
        resname_map=polyethylene_resname_map,
    )
    mol = _load_sdf(valid_path)[0]
    mol.GetAtomWithIdx(1).SetProp("mupt_segment_label", "other-segment")
    prepare_mupt_sdf_atom_props(mol)
    _write_mol(malformed_path, mol)

    with pytest.raises(ValueError, match="inconsistent SEGMENT identity"):
        primitive_from_mupt_sdf(malformed_path)


def test_primitive_from_mupt_sdf_rejects_conflicting_residue_labels(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    """Atoms sharing a MuPT residue index must agree on RESIDUE label."""
    valid_path = tmp_path / "valid.mupt.sdf"
    malformed_path = tmp_path / "bad-residue.mupt.sdf"
    write_primitive_to_sdf(
        single_polyethylene_2mer,
        valid_path,
        resname_map=polyethylene_resname_map,
    )
    mol = _load_sdf(valid_path)[0]
    first_residue_index = mol.GetAtomWithIdx(0).GetProp("mupt_residue_index")
    same_residue_atom = next(
        atom
        for atom in mol.GetAtoms()
        if atom.GetIdx() != 0
        and atom.HasProp("mupt_residue_index")
        and atom.GetProp("mupt_residue_index") == first_residue_index
    )
    same_residue_atom.SetProp("mupt_residue_label", "other-residue")
    prepare_mupt_sdf_atom_props(mol)
    _write_mol(malformed_path, mol)

    with pytest.raises(ValueError, match="conflicting RESIDUE labels"):
        primitive_from_mupt_sdf(malformed_path)
