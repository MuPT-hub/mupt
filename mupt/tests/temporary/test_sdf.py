"""Tests for temporary MuPT SDF interoperability export."""

import subprocess
import sys
import warnings

import numpy as np
import pytest
from anytree import PreOrderIter
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SDMolSupplier, SDWriter

import mupt.interfaces.rdkit.exporters as rdkit_exporters
import mupt.temporary.sdf as temporary_sdf
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole
from mupt.temporary.sdf import (
    MUPT_SDF_ATOM_PROPS,
    iter_primitives_from_mupt_sdf,
    primitive_from_mupt_sdf,
    prepare_mupt_sdf_atom_props,
    write_primitive_to_mupt_sdf,
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
assert callable(write_primitive_to_sdf)
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


def test_temporary_package_reexports_mupt_sdf_writer(
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
    for path in [tmp_path / "no-suffix", tmp_path / "sdf-suffix.sdf"]:
        with pytest.warns(
            UserWarning,
            match="MuPT temporary SDF files use the '.mupt.sdf' suffix",
        ):
            records = write_primitive_to_sdf(
                single_polyethylene_2mer,
                path,
                resname_map=polyethylene_resname_map,
            )

        assert records == 1
        assert _mupt_sdf_path(path).exists()

    path = tmp_path / "mupt-suffix.mupt.sdf"
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        records = write_primitive_to_sdf(
            single_polyethylene_2mer,
            path,
            resname_map=polyethylene_resname_map,
        )

    assert records == 1
    assert not caught_warnings
    assert _mupt_sdf_path(path).exists()


def test_write_primitive_to_sdf_preserves_existing_file_after_stream_failure(
    tmp_path,
    monkeypatch,
):
    """Failed streaming exports must not replace a complete existing SDF."""
    final_path = tmp_path / "partial.mupt.sdf"
    existing_contents = "existing complete file\n"
    final_path.write_text(existing_contents)

    def failing_mols(*args, **kwargs):
        yield Chem.MolFromSmiles("C")
        raise RuntimeError("export failed")

    monkeypatch.setattr(temporary_sdf, "primitive_to_rdkit_mols", failing_mols)

    with pytest.raises(RuntimeError, match="export failed"):
        write_primitive_to_sdf(
            Primitive(label="temporary", role=PrimitiveRole.UNIVERSE),
            tmp_path / "partial.sdf",
            resname_map={},
        )

    assert final_path.read_text() == existing_contents


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


def test_iter_primitives_from_mupt_sdf_yields_one_segment_per_record(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    """The streaming importer yields SEGMENT roots without building a UNIVERSE."""
    sdf_path = tmp_path / "streaming-multi.mupt.sdf"
    records = write_primitive_to_sdf(
        multi_polyethylene_system,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )

    segments = list(iter_primitives_from_mupt_sdf(sdf_path))

    assert len(segments) == records == len(multi_polyethylene_system.children)
    assert [segment.role for segment in segments] == [PrimitiveRole.SEGMENT] * records
    assert all(segment.parent is None for segment in segments)


def test_iter_primitives_from_mupt_sdf_uses_forward_supplier(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
    monkeypatch,
):
    """The streaming importer uses RDKit's forward-only SDF supplier."""
    sdf_path = tmp_path / "forward-supplier.mupt.sdf"
    supplier_calls = []
    original_supplier = temporary_sdf.ForwardSDMolSupplier

    def supplier_spy(*args, **kwargs):
        supplier_calls.append((args, kwargs))
        return original_supplier(*args, **kwargs)

    write_primitive_to_sdf(
        single_polyethylene_2mer,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    monkeypatch.setattr(temporary_sdf, "ForwardSDMolSupplier", supplier_spy)

    segments = list(iter_primitives_from_mupt_sdf(sdf_path))

    assert len(segments) == 1
    assert len(supplier_calls) == 1
    assert supplier_calls[0][1]["sanitize"] is False
    assert supplier_calls[0][1]["removeHs"] is False


def test_iter_primitives_from_mupt_sdf_streaming_discard_preserves_counts(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    """Streaming consumers can aggregate segment data without a materialized root."""
    sdf_path = tmp_path / "streaming-discard.mupt.sdf"
    first_records = write_primitive_to_sdf(
        multi_polyethylene_system,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    source_mols = _load_sdf(sdf_path)

    streamed_atom_counts = []
    streamed_bond_counts = []
    streamed_segment_labels = []
    for segment in iter_primitives_from_mupt_sdf(sdf_path):
        assert segment.role is PrimitiveRole.SEGMENT
        assert segment.parent is None
        streamed_atom_counts.append(segment.num_atoms)
        streamed_bond_counts.append(_total_internal_bonds(segment))
        streamed_segment_labels.append(segment.label)
        del segment

    assert len(streamed_atom_counts) == first_records
    assert streamed_atom_counts == [mol.GetNumAtoms() for mol in source_mols]
    assert streamed_bond_counts == [mol.GetNumBonds() for mol in source_mols]
    assert streamed_segment_labels == [segment.label for segment in multi_polyethylene_system.children]


def test_streamed_segment_attached_to_universe_reexports_mupt_props(
    tmp_path,
    single_polyethylene_3mer,
    polyethylene_resname_map,
):
    """A streamed segment remains exportable after attachment to a new root."""
    first_path = tmp_path / "streamed-first.mupt.sdf"
    second_path = tmp_path / "streamed-second.mupt.sdf"
    write_primitive_to_sdf(
        single_polyethylene_3mer,
        first_path,
        resname_map=polyethylene_resname_map,
    )
    first_mol = _load_sdf(first_path)[0]
    streamed_segment = next(iter_primitives_from_mupt_sdf(first_path))
    universe = Primitive(label="temporary", role=PrimitiveRole.UNIVERSE)

    universe.attach_child(streamed_segment)
    second_records = write_primitive_to_sdf(
        universe,
        second_path,
        resname_map=polyethylene_resname_map,
    )
    second_mol = _load_sdf(second_path)[0]

    assert second_records == 1
    assert second_mol.GetNumAtoms() == first_mol.GetNumAtoms()
    assert second_mol.GetNumBonds() == first_mol.GetNumBonds()
    assert _atom_mupt_props(second_mol) == _atom_mupt_props(first_mol)


def test_iter_primitives_from_mupt_sdf_sanitizes_with_mupt_helper(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
    monkeypatch,
):
    """sanitize=True uses MuPT's sanitizer, not RDKit supplier sanitization."""
    sdf_path = tmp_path / "sanitize-path.mupt.sdf"
    sanitizer_calls = []

    def sanitize_spy(mol):
        sanitizer_calls.append(mol.GetNumAtoms())
        return mol

    write_primitive_to_sdf(
        single_polyethylene_2mer,
        sdf_path,
        resname_map=polyethylene_resname_map,
    )
    monkeypatch.setattr(temporary_sdf, "sanitized_mol", sanitize_spy)

    segments = list(iter_primitives_from_mupt_sdf(sdf_path, sanitize=True))

    assert len(segments) == 1
    assert sanitizer_calls == [segments[0].num_atoms]


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
