'''
Tests for ensuring export from MuPT to RDKit preserves chemical information and metadata,
and does not export systems which cannot be interpreted as all-atom molecules
'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

import pytest
from anytree import PreOrderIter
from rdkit import Chem

from mupt.chemistry import ELEMENTS
from mupt.interfaces.rdkit import AllAtomRDKitExportStrategy, primitive_to_rdkit_mols
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole

# TODO: test chemical info (e.g. charge, isotope, etc.) is preserved on atoms

# TODO: test metadata transfer
# for atom in mol.GetAtoms():
#     atom.SetDoubleProp('mass', ptab.GetAtomicWeight(atom.GetAtomicNum()))
# mol.SetProp('name', 'benzoic_acid')
# mol.SetBoolProp('is_aromatic', True)


def _rdkit_mols(*args, **kwargs):
    """Materialize the streaming exporter for small unit-test fixtures."""
    return list(primitive_to_rdkit_mols(*args, **kwargs))


def _count_internal_connections(root: Primitive) -> int:
    return sum(
        len(node.internal_connections)
        for node in PreOrderIter(root)
        if not node.is_leaf
    )


def _universe_from_residue(residue: Primitive) -> Primitive:
    """Wrap a repeat-unit primitive in the role-aware SAAMR hierarchy."""
    residue.role = PrimitiveRole.RESIDUE
    for atom in residue.leaves:
        atom.role = PrimitiveRole.PARTICLE
    segment = Primitive(label="chain", role=PrimitiveRole.SEGMENT)
    segment.attach_child(residue)
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(segment)
    return universe


def test_primitive_to_rdkit_mols_returns_one_mol_per_segment(
    multi_polyethylene_system,
    polyethylene_resname_map,
):
    mols = _rdkit_mols(multi_polyethylene_system, polyethylene_resname_map)

    assert len(mols) == 10


def test_primitive_to_rdkit_mols_preserves_atom_count(
    single_polyethylene_3mer,
    polyethylene_resname_map,
):
    mols = _rdkit_mols(single_polyethylene_3mer, polyethylene_resname_map)

    assert len(mols) == 1
    assert mols[0].GetNumAtoms() == len(single_polyethylene_3mer.leaves)


def test_primitive_to_rdkit_mols_preserves_bond_count(
    depth4_bonded_system,
    polyethylene_resname_map,
):
    mols = _rdkit_mols(depth4_bonded_system, polyethylene_resname_map)

    assert len(mols) == 1
    assert mols[0].GetNumBonds() == _count_internal_connections(depth4_bonded_system)


def test_primitive_to_rdkit_mols_sets_pdb_residue_info(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    mol = _rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]

    for atom in mol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        assert pdb_info is not None
        assert pdb_info.GetChainId() == "A"
        assert pdb_info.GetResidueName().strip() in set(polyethylene_resname_map.values())
        assert atom.GetProp("chain_id") == "A"
        assert atom.GetProp("residue_name") in set(polyethylene_resname_map.values())


def test_primitive_to_rdkit_mols_calls_strategy_validate(
    single_polyethylene_2mer,
    polyethylene_resname_map,
):
    class TrackingStrategy(AllAtomRDKitExportStrategy):
        def __init__(self):
            super().__init__()
            self.validated = False

        def validate(self, root):
            self.validated = True
            super().validate(root)

    strategy = TrackingStrategy()

    _rdkit_mols(
        single_polyethylene_2mer,
        polyethylene_resname_map,
        strategy=strategy,
    )

    assert strategy.validated


def test_primitive_to_rdkit_mols_rejects_empty_segment():
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(Primitive(label="empty", role=PrimitiveRole.SEGMENT))

    with pytest.raises(ValueError, match="contains no RESIDUE"):
        _rdkit_mols(universe, {})


def test_primitive_to_rdkit_mols_rejects_empty_residue():
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    segment.attach_child(Primitive(label="empty", role=PrimitiveRole.RESIDUE))
    universe.attach_child(segment)

    with pytest.raises(ValueError, match="no PARTICLE leaves"):
        _rdkit_mols(universe, {"empty": "EMP"})


def test_primitive_to_rdkit_mols_rejects_unassigned_root():
    universe = Primitive(label="universe")
    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    residue = Primitive(label="res", role=PrimitiveRole.RESIDUE)
    atom = Primitive(label="He", element=ELEMENTS[2], role=PrimitiveRole.PARTICLE)
    universe.attach_child(segment)
    segment.attach_child(residue)
    residue.attach_child(atom)

    with pytest.raises(ValueError, match="UNIVERSE"):
        _rdkit_mols(universe, {"res": "RES"})


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
    universe = _universe_from_residue(residue)

    mols = _rdkit_mols(universe, {label: "UNK"})

    assert len(mols) == 1
    assert mols[0].GetNumAtoms() == len(residue.leaves) + len(universe.children[0].external_connectors)
    assert mols[0].GetNumBonds() == _count_internal_connections(residue) + len(universe.children[0].external_connectors)


def test_primitive_to_rdkit_mols_preserves_valid_thiophene_chemistry():
    """Issue #31: heteroaromatic thiophene exports remain RDKit-sanitizable."""
    residue = primitive_from_smiles(
        "*-[C:1]1=C-C=[C:2](-S-1)-*",
        ensure_explicit_Hs=True,
        label="mid_thiophene",
    )
    universe = _universe_from_residue(residue)

    mol = _rdkit_mols(universe, {"mid_thiophene": "THI"})[0]
    Chem.SanitizeMol(Chem.Mol(mol))

    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()].count(0) == 2
    assert [atom.GetSymbol() for atom in mol.GetAtoms()].count("S") == 1
    assert any(bond.GetIsAromatic() for bond in mol.GetBonds())
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    assert sulfur.GetTotalValence() == 2
