"""Tests for temporary MuPT SDF interoperability export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from rdkit.Chem.rdmolfiles import SDMolSupplier

from mupt.interfaces.rdkit import write_primitive_to_mupt_sdf
from mupt.temporary.sdf import MUPT_SDF_ATOM_PROPS, write_primitive_to_sdf


def _load_sdf(path):
    return [mol for mol in SDMolSupplier(str(path), removeHs=False, sanitize=False) if mol is not None]


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
