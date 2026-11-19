"""Tests that tetrahedral stereo survives RDKit -> MuPT -> RDKit round trips"""

import pytest
from rdkit import Chem

from mupt.interfaces.smiles import primitive_from_smiles
from mupt.interfaces.rdkit import rdkit_mol_from_primitive
from mupt.interfaces.rdkit.components import STEREO_METADATA_KEYS
from mupt.mupr.connection import Connector, IncompatibleConnectorError, AttachmentPoint


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


@pytest.mark.parametrize(
    "smiles",
    ["C[C@@H](c1ccccc1)C[C@H](Cl)CC", "C[C@H](c1ccccc1)C[C@@H](Cl)CC", "N[C@@H](C)C(=O)O"],
)
@pytest.mark.parametrize("embed", [False, True])
def test_stereo_round_trip(smiles: str, embed: bool) -> None:
    prim = primitive_from_smiles(smiles, embed_positions=embed, embed_random_seed=0)
    assert canonical(rdkit_mol_from_primitive(prim.copy())) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_stereo_metadata_not_exported_as_props() -> None:
    mol = rdkit_mol_from_primitive(primitive_from_smiles("N[C@@H](C)C(=O)O"))
    props = {name for atom in mol.GetAtoms() for name in atom.GetPropNames()}
    props |= {name for bond in mol.GetBonds() for name in bond.GetPropNames()}
    assert not (props & STEREO_METADATA_KEYS)


def test_free_ports_export_as_labelled_linkers() -> None:
    mol = rdkit_mol_from_primitive(primitive_from_smiles("O[*:3]"))
    assert [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0] == [3]


def test_bond_to_can_skip_alignment() -> None:
    def offset_connector() -> Connector:
        return Connector(anchor=AttachmentPoint(position=[0.0, 0.0, 0.0]), linker=AttachmentPoint(position=[1.0, 0.0, 0.0]))

    first, second = offset_connector(), offset_connector()
    with pytest.raises(IncompatibleConnectorError):
        first.neighbor = second
    first.bond_to(second, check_alignment=False)
    assert first.neighbor is second and second.neighbor is first


@pytest.mark.parametrize("smiles", ["*CCO*", "*CC(C)(C)C*", "[*:1]C1C([*:2])C([*:3])C1[*:4]"])
def test_embedding_with_nearby_linkers(smiles: str) -> None:
    # RDKit distance geometry fails on bare wild-type atoms 4 bonds apart
    prim = primitive_from_smiles(smiles, embed_random_seed=0)
    assert all(atom.shape is not None for atom in prim.children)
