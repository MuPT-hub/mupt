'''Test that no information is lost when converting from and then back to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

import json

import pytest
from rdkit.Chem.rdchem import Atom, AtomPDBResidueInfo, Mol, RWMol
from rdkit.Chem.rdmolfiles import MolFromSmiles, SDWriter
from rdkit.Chem.rdmolops import AddHs

from mupt.mupr.connection import TraversalDirection
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.interfaces.rdkit import importers, primitive_from_mupt_sdf, primitive_to_rdkit_mols

# TODO: test chemical info (e.g. charge, isotope, etc.) is preserved on atoms

@pytest.fixture(scope='function')
def mol() -> Mol:
    '''A simple test molecule with nontrivial chemical features'''
    rdmol = MolFromSmiles('[NH3+]Cc1c(C#N)c(C-[*:34])ccc1C(-[O-])=O')
    rdmol = AddHs(rdmol)
    # conf_id = EmbedMolecule(mol)

    return rdmol

@pytest.fixture(scope='function')
def primitive(mol : Mol) -> Primitive:
    return importers.primitive_from_rdkit(mol)


def test_valences_permissible(primitive : Primitive) -> None:
    '''Check that chemical valences for all atomic Primitives are among those allowable for their assigned element'''
    assert all(atomprim.is_atom for atomprim in primitive.leaves)


def test_primitive_from_rdkit_defaults_to_saamr_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 1
    segment = primitive.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 1
    residue = segment.children[0]
    assert residue.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in residue.children)


def test_primitive_from_rdkit_preserves_legacy_denest_default(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_rdkit_legacy_denest_uses_residue_role(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(
        mol,
        denest=True,
        role=PrimitiveRole.UNIVERSE,
        residue_role=PrimitiveRole.SEGMENT,
    )

    assert primitive.role == PrimitiveRole.SEGMENT


def test_primitive_from_rdkit_denest_false_returns_saamr_hierarchy(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 1
    segment = primitive.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 1
    residue = segment.children[0]
    assert residue.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in residue.children)


def test_primitive_from_rdkit_denest_false_preserves_root_metadata() -> None:
    mol = MolFromSmiles("CC.CC")
    mol.SetProp("mupt_test_root", "preserved")

    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.metadata["mupt_test_root"] == "preserved"
    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 2


def test_primitive_from_rdkit_does_not_reclassify_external_prefixed_metadata() -> None:
    mol = MolFromSmiles("CC")
    mol.SetProp("mupt_root_metadata_external", "segment")

    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.metadata["mupt_root_metadata_external"] == "segment"
    assert "external" not in primitive.metadata
    assert primitive.children[0].metadata["mupt_root_metadata_external"] == "segment"


def test_primitive_from_rdkit_fallback_residue_key_includes_chain_id() -> None:
    editable = RWMol()
    for chain_id in ("A", "B"):
        atom = Atom(6)
        atom.SetMonomerInfo(
            AtomPDBResidueInfo(
                atomName=" C  ",
                residueName="RES",
                residueNumber=1,
                chainId=chain_id,
            )
        )
        editable.AddAtom(atom)
    editable.AddBond(0, 1)

    primitive = importers.primitive_from_rdkit(Mol(editable), denest=False)

    assert len(primitive.children[0].children) == 2


def test_primitive_from_rdkit_fallback_residue_key_includes_insertion_code() -> None:
    editable = RWMol()
    for insertion_code in ("A", "B"):
        atom = Atom(6)
        pdb_info = AtomPDBResidueInfo(
            atomName=" C  ",
            residueName="RES",
            residueNumber=1,
            chainId="A",
        )
        pdb_info.SetInsertionCode(insertion_code)
        atom.SetMonomerInfo(pdb_info)
        editable.AddAtom(atom)
    editable.AddBond(0, 1)

    primitive = importers.primitive_from_rdkit(Mol(editable), denest=False)

    assert len(primitive.children[0].children) == 2

    exported = primitive_to_rdkit_mols(primitive, {"RES": "RES"})[0]
    insertion_codes = {
        atom.GetPDBResidueInfo().GetInsertionCode()
        for atom in exported.GetAtoms()
    }
    reimported = importers.primitive_from_rdkit(exported, denest=False)
    reimported_insertion_codes = {
        residue.metadata["pdb_insertion_code"]
        for residue in reimported.children[0].children
    }

    assert insertion_codes == {"A", "B"}
    assert reimported_insertion_codes == {"A", "B"}


def test_primitive_from_rdkit_denest_false_preserves_linker_connectors() -> None:
    mol = MolFromSmiles("*-[C:1]-[C:2]-*")

    primitive = importers.primitive_from_rdkit(mol, denest=False)
    segment = primitive.children[0]
    residue = segment.children[0]

    assert len(residue.external_connectors) == 2
    assert len(segment.external_connectors) == 2
    residue_directions = {
        direction
        for conn_handle in residue.external_connectors
        for direction in residue.fetch_connector(conn_handle).anchor.attachables
        if isinstance(direction, TraversalDirection)
    }
    segment_directions = {
        direction
        for conn_handle in segment.external_connectors
        for direction in segment.fetch_connector(conn_handle).anchor.attachables
        if isinstance(direction, TraversalDirection)
    }

    assert residue_directions == {TraversalDirection(1), TraversalDirection(2)}
    assert segment_directions == {TraversalDirection(1), TraversalDirection(2)}


def test_primitive_from_smiles_defaults_to_residue_particle_roles() -> None:
    primitive = primitive_from_smiles("CC", ensure_explicit_Hs=True)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_rdkit_accepts_explicit_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(
        mol,
        denest=False,
        role=PrimitiveRole.UNIVERSE,
        residue_role=PrimitiveRole.RESIDUE,
        atom_role=PrimitiveRole.PARTICLE,
    )

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert primitive.children[0].role == PrimitiveRole.SEGMENT
    assert primitive.children[0].children[0].role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.leaves)


def test_rdkit_export_import_preserves_saamr_hierarchy(
    single_polyethylene_2mer,
    polyethylene_resname_map,
) -> None:
    rdkit_mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]
    reconstructed = importers.primitive_from_rdkit(rdkit_mol, denest=False)

    assert reconstructed.role == PrimitiveRole.UNIVERSE
    assert len(reconstructed.children) == 1
    segment = reconstructed.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 2
    assert [residue.label for residue in segment.children] == ["head", "tail"]
    assert all(residue.role == PrimitiveRole.RESIDUE for residue in segment.children)
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in reconstructed.leaves)
    assert len(reconstructed.leaves) == len(single_polyethylene_2mer.leaves)


def _write_sdf(path, mols) -> None:
    writer = SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()


def test_primitive_from_mupt_sdf_returns_universe_for_single_sdf(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
) -> None:
    sdf_path = tmp_path / "chain.sdf"
    _write_sdf(sdf_path, primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map))

    reconstructed = primitive_from_mupt_sdf(sdf_path)

    assert reconstructed.role == PrimitiveRole.UNIVERSE
    assert len(reconstructed.children) == 1
    assert reconstructed.children[0].role == PrimitiveRole.SEGMENT
    assert [residue.role for residue in reconstructed.children[0].children] == [
        PrimitiveRole.RESIDUE,
        PrimitiveRole.RESIDUE,
    ]
    assert len(reconstructed.leaves) == len(single_polyethylene_2mer.leaves)


def test_primitive_from_mupt_sdf_accepts_multiple_sdf_paths(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
) -> None:
    sdf_paths = []
    mols = primitive_to_rdkit_mols(multi_polyethylene_system, polyethylene_resname_map)
    for idx, mol in enumerate(mols[:2]):
        sdf_path = tmp_path / f"chain_{idx}.sdf"
        _write_sdf(sdf_path, [mol])
        sdf_paths.append(sdf_path)

    reconstructed = primitive_from_mupt_sdf(sdf_paths)

    assert reconstructed.role == PrimitiveRole.UNIVERSE
    assert len(reconstructed.children) == 2
    assert all(segment.role == PrimitiveRole.SEGMENT for segment in reconstructed.children)


def test_primitive_from_mupt_sdf_accepts_multirecord_sdf(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
) -> None:
    sdf_path = tmp_path / "chains.sdf"
    _write_sdf(sdf_path, primitive_to_rdkit_mols(multi_polyethylene_system, polyethylene_resname_map)[:2])

    reconstructed = primitive_from_mupt_sdf(sdf_path)

    assert len(reconstructed.children) == 2


def test_primitive_from_mupt_sdf_rejects_mixed_root_metadata(
    tmp_path,
    multi_polyethylene_system,
    polyethylene_resname_map,
) -> None:
    sdf_path = tmp_path / "chains.sdf"
    multi_polyethylene_system.metadata["mupt_test_root"] = "shared"
    mols = primitive_to_rdkit_mols(multi_polyethylene_system, polyethylene_resname_map)[:2]
    mols[1].SetProp("mupt_root_metadata_key_0", "conflicting")
    mols[1].SetProp("mupt_root_metadata_value_0", "metadata")

    _write_sdf(sdf_path, mols)

    with pytest.raises(ValueError, match="UNIVERSE metadata"):
        primitive_from_mupt_sdf(sdf_path)


def test_primitive_from_mupt_sdf_rejects_inconsistent_hierarchy_prefix(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
) -> None:
    sdf_path = tmp_path / "chain.sdf"
    mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]
    atom = next(candidate for candidate in mol.GetAtoms() if candidate.GetAtomicNum() != 0)
    path_entries = json.loads(atom.GetProp("mupt_hierarchy_path_json"))
    path_entries[0]["label"] = "other_segment"
    atom.SetProp("mupt_hierarchy_path_json", json.dumps(path_entries, separators=(",", ":")))

    _write_sdf(sdf_path, [mol])

    with pytest.raises(ValueError, match="path prefix"):
        primitive_from_mupt_sdf(sdf_path)


def test_primitive_from_mupt_sdf_preserves_intermediate_nodes(tmp_path) -> None:
    residue = primitive_from_smiles(
        "CC",
        label="ethane",
        ensure_explicit_Hs=True,
        embed_positions=True,
    )
    group = Primitive(label="functional_group")
    group.attach_child(residue)
    segment = Primitive(label="chain", role=PrimitiveRole.SEGMENT)
    segment.attach_child(group)
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(segment)
    residue.role = PrimitiveRole.RESIDUE
    for atom in residue.leaves:
        atom.role = PrimitiveRole.PARTICLE
    sdf_path = tmp_path / "grouped.sdf"
    _write_sdf(sdf_path, primitive_to_rdkit_mols(universe, {"ethane": "EAN"}))

    reconstructed = primitive_from_mupt_sdf(sdf_path)

    segment = reconstructed.children[0]
    assert segment.children[0].label == "functional_group"
    assert segment.children[0].role == PrimitiveRole.UNASSIGNED
    assert segment.children[0].children[0].role == PrimitiveRole.RESIDUE


def test_primitive_from_mupt_sdf_rejects_missing_serialization_metadata(tmp_path) -> None:
    sdf_path = tmp_path / "plain.sdf"
    _write_sdf(sdf_path, [MolFromSmiles("CC")])

    with pytest.raises(ValueError, match="serialization metadata"):
        primitive_from_mupt_sdf(sdf_path)


def test_primitive_from_mupt_sdf_can_skip_bond_reconstruction(
    tmp_path,
    single_polyethylene_2mer,
    polyethylene_resname_map,
) -> None:
    sdf_path = tmp_path / "chain.sdf"
    _write_sdf(sdf_path, primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map))

    reconstructed = primitive_from_mupt_sdf(
        sdf_path,
        reconstruct_bonds=False,
        reconstruct_shapes=False,
    )

    assert reconstructed.role == PrimitiveRole.UNIVERSE
    assert len(reconstructed.children[0].children) == 2
    assert len(reconstructed.leaves) == len(single_polyethylene_2mer.leaves)
    assert all(len(node.internal_connections) == 0 for node in reconstructed.descendants)
