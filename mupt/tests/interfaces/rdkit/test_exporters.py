'''
Tests for ensuring export from MuPT to RDKit preserves chemical information and metadata,
and does not export systems which cannot be interpreted as all-atom molecules
'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

import pytest
from anytree import PreOrderIter
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from mupt.chemistry import ELEMENTS
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.interfaces.rdkit import importers, AllAtomRDKitExportStrategy, primitive_to_rdkit_mols
from mupt.interfaces.rdkit.exporters import _pdb_chain_and_resid

# TODO: test chemical info (e.g. charge, isotope, etc.) is preserved on atoms

# TODO: test metadata transfer
# for atom in mol.GetAtoms():
#     atom.SetDoubleProp('mass', ptab.GetAtomicWeight(atom.GetAtomicNum()))
# mol.SetProp('name', 'benzoic_acid')
# mol.SetBoolProp('is_aromatic', True)


def _count_internal_connections(root: Primitive) -> int:
    return sum(
        len(node.internal_connections)
        for node in PreOrderIter(root)
        if not node.is_leaf
    )


def test_primitive_to_rdkit_mols_returns_one_mol_per_segment(
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    mols = primitive_to_rdkit_mols(multi_polyethylene_system, polyethylene_resname_map)

    assert len(mols) == 10


def test_primitive_to_rdkit_mols_preserves_atom_count(
    single_polyethylene_3mer,
    polyethylene_resname_map,
):
    mols = primitive_to_rdkit_mols(single_polyethylene_3mer, polyethylene_resname_map)

    assert len(mols) == 1
    assert mols[0].GetNumAtoms() == len(single_polyethylene_3mer.leaves)


def test_primitive_to_rdkit_mols_preserves_bond_count(
    depth4_bonded_system,
    polyethylene_resname_map,
):
    mols = primitive_to_rdkit_mols(depth4_bonded_system, polyethylene_resname_map)

    assert len(mols) == 1
    assert mols[0].GetNumBonds() == _count_internal_connections(depth4_bonded_system)


def test_primitive_to_rdkit_mols_preserves_inter_residue_bond(
    depth4_bonded_system,
    polyethylene_resname_map,
):
    mol = primitive_to_rdkit_mols(depth4_bonded_system, polyethylene_resname_map)[0]
    cross_residue_bonds = [
        bond for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetIntProp("residue_id") != bond.GetEndAtom().GetIntProp("residue_id")
    ]

    assert len(cross_residue_bonds) == 1
    assert cross_residue_bonds[0].GetBeginAtom().GetSymbol() == "C"
    assert cross_residue_bonds[0].GetEndAtom().GetSymbol() == "C"


def test_primitive_to_rdkit_mols_preserves_bond_metadata_from_both_connectors(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    bond_node = next(node for node in PreOrderIter(single_polyethylene_2mer) if node.internal_connections)
    conn_ref1, conn_ref2 = tuple(next(iter(bond_node.internal_connections)))
    bond_node.fetch_connector_on_child(conn_ref1).metadata["mupt_test_conn1"] = "left"
    bond_node.fetch_connector_on_child(conn_ref2).metadata["mupt_test_conn2"] = "right"

    mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]
    metadata_bonds = [bond for bond in mol.GetBonds() if bond.HasProp("mupt_test_conn1")]

    assert len(metadata_bonds) == 1
    assert metadata_bonds[0].GetProp("mupt_test_conn1") == "left"
    assert metadata_bonds[0].GetProp("mupt_test_conn2") == "right"


def test_primitive_to_rdkit_mols_preserves_root_metadata_on_round_trip(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    single_polyethylene_2mer.metadata["mupt_test_shared"] = "root"
    single_polyethylene_2mer.children[0].metadata["mupt_test_shared"] = "segment"

    mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]
    reconstructed = importers.primitive_from_rdkit(mol, denest=False)

    assert reconstructed.metadata["mupt_test_shared"] == "root"
    assert reconstructed.children[0].metadata["mupt_test_shared"] == "segment"


def test_primitive_to_rdkit_mols_rejects_root_transport_metadata_collision(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    single_polyethylene_2mer.children[0].metadata["mupt_root_metadata_count"] = 1

    with pytest.raises(ValueError, match="reserved"):
        primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)


def test_primitive_to_rdkit_mols_preserves_external_connectors_on_round_trip():
    residue = primitive_from_smiles("*-[C:1]-[C:2]-*", label="mid", ensure_explicit_Hs=True)
    segment = Primitive(label="chain", role=PrimitiveRole.SEGMENT)
    segment.attach_child(residue)
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(segment)

    mol = primitive_to_rdkit_mols(universe, {"mid": "MID"})[0]
    reconstructed = importers.primitive_from_rdkit(mol, denest=False)
    linker_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]

    assert mol.GetNumAtoms() == len(residue.leaves) + 2
    assert all(atom.GetPDBResidueInfo() is not None for atom in linker_atoms)
    assert all(atom.GetProp("chain_id") == "A" for atom in linker_atoms)
    assert len(reconstructed.children[0].external_connectors) == 2
    assert len(reconstructed.children[0].children[0].external_connectors) == 2


def test_primitive_to_rdkit_mols_sets_pdb_residue_info(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]

    for atom in mol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        assert pdb_info is not None
        assert pdb_info.GetChainId() == "A"
        assert pdb_info.GetResidueName().strip() in set(polyethylene_resname_map.values())
        assert atom.GetProp("chain_id") == "A"
        assert atom.GetProp("residue_name") in set(polyethylene_resname_map.values())


def test_primitive_to_rdkit_mols_numbers_residues_across_segments(
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    mols = primitive_to_rdkit_mols(multi_polyethylene_system, polyethylene_resname_map)
    residue_ids = [
        mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueNumber()
        for mol in mols
    ]
    expected_residue_ids = []
    next_residue_id = 1
    for segment in multi_polyethylene_system.children:
        expected_residue_ids.append(next_residue_id)
        next_residue_id += len(segment.children)

    assert residue_ids == expected_residue_ids
    assert {mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetChainId() for mol in mols} == {"A"}


def test_pdb_chain_and_resid_uses_9999_residue_buckets() -> None:
    assert _pdb_chain_and_resid(0) == ("A", 1)
    assert _pdb_chain_and_resid(9998) == ("A", 9999)
    assert _pdb_chain_and_resid(9999) == ("B", 1)


def test_explicit_rdkit_strategy_produces_same_result(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    default_mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]
    strategy = AllAtomRDKitExportStrategy()
    explicit_mol = primitive_to_rdkit_mols(
        single_polyethylene_2mer,
        polyethylene_resname_map,
        strategy=strategy,
    )[0]

    assert explicit_mol.GetNumAtoms() == default_mol.GetNumAtoms()
    assert explicit_mol.GetNumBonds() == default_mol.GetNumBonds()


def test_rdkit_strategy_rejects_unassigned_root():
    universe = Primitive(label="universe")
    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    residue = Primitive(label="res", role=PrimitiveRole.RESIDUE)
    atom = Primitive(label="He", element=ELEMENTS[2], role=PrimitiveRole.PARTICLE)
    universe.attach_child(segment)
    segment.attach_child(residue)
    residue.attach_child(atom)

    with pytest.raises(ValueError, match="UNIVERSE"):
        primitive_to_rdkit_mols(universe, {"res": "RES"})


@pytest.mark.parametrize(
    "label,smiles",
    [
        ("mid_thiophene", "*-[C:1]1=C-C=[C:2](-S-1)-*"),
        ("mid_pyrrole", "*-[C:1]1=C-C=[C:2](-[NH]-1)-*"),
        (
            "mid_pyromellitimide",
            "*-[N:1]1C(=O)c2c(C(=O)1)cc3c(c2)C(=O)[N:2](C(=O)3)-*",
        ),
    ],
)
def test_primitive_to_rdkit_mols_exports_heterocyclic_aromatics(label, smiles):
    """Regression coverage for issue #31 on the role-aware exporter path."""
    residue = primitive_from_smiles(smiles, ensure_explicit_Hs=True, label=label)
    segment = Primitive(label="chain", role=PrimitiveRole.SEGMENT)
    segment.attach_child(residue)
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(segment)

    mols = primitive_to_rdkit_mols(universe, {label: "UNK"})

    assert len(mols) == 1
    assert mols[0].GetNumAtoms() == len(residue.leaves) + len(segment.external_connectors)
    assert mols[0].GetNumBonds() == _count_internal_connections(residue) + len(segment.external_connectors)


def test_primitive_to_rdkit_mols_preserves_valid_thiophene_chemistry():
    """The role-aware exporter keeps heterocyclic chemistry chemically valid."""
    residue = primitive_from_smiles(
        "*-[C:1]1=C-C=[C:2](-S-1)-*",
        ensure_explicit_Hs=True,
        label="mid_thiophene",
    )
    segment = Primitive(label="chain", role=PrimitiveRole.SEGMENT)
    segment.attach_child(residue)
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(segment)

    for atom in residue.leaves:
        atom.check_valence()

    mol = primitive_to_rdkit_mols(universe, {"mid_thiophene": "THI"})[0]
    Chem.SanitizeMol(Chem.Mol(mol))

    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()].count(0) == 2
    assert [atom.GetSymbol() for atom in mol.GetAtoms()].count("S") == 1
    assert sorted(bond.GetBondType() for bond in mol.GetBonds()) == sorted(
        [BondType.DOUBLE, BondType.DOUBLE] + [BondType.SINGLE] * 7
    )
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    assert sulfur.GetTotalValence() == 2
