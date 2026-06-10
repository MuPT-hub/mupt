"""Fast tests for all-atom DPD builder utilities."""

import builtins
import importlib
import sys

import numpy as np
from periodictable import elements

from mupt.chemistry.core import BondType
from mupt.mupr.connection import AttachmentPoint, Connector
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole


def _bond_connector(anchor: str, linker: str, label: str) -> Connector:
    return Connector(
        anchor=AttachmentPoint({anchor}),
        linker=AttachmentPoint({linker}),
        bondtype=BondType.SINGLE,
        label=label,
    )


def _tiny_saamr_hierarchy() -> tuple[Primitive, list[Primitive]]:
    """Return universe -> segment -> residue -> H-C-H with two bonds."""

    h1 = Primitive(label="H1", element=elements.H, role=PrimitiveRole.PARTICLE)
    c = Primitive(label="C", element=elements.C, role=PrimitiveRole.PARTICLE)
    h2 = Primitive(label="H2", element=elements.H, role=PrimitiveRole.PARTICLE)

    h1_conn = h1.register_connector(_bond_connector("H", "C", "h"))
    c_left_conn = c.register_connector(_bond_connector("C", "H", "left"))
    c_right_conn = c.register_connector(_bond_connector("C", "H", "right"))
    h2_conn = h2.register_connector(_bond_connector("H", "C", "h"))

    residue = Primitive(label="res", role=PrimitiveRole.RESIDUE)
    h1_handle = residue.attach_child(h1)
    c_handle = residue.attach_child(c)
    h2_handle = residue.attach_child(h2)
    residue.connect_children(h1_handle, h1_conn, c_handle, c_left_conn)
    residue.connect_children(h2_handle, h2_conn, c_handle, c_right_conn)

    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    segment.attach_child(residue)

    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(segment)
    return universe, [h1, c, h2]


def test_imports_public_symbols_without_hoomd_or_openff(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "hoomd" or name.startswith("hoomd.") or name == "openff" or name.startswith("openff."):
            raise AssertionError(f"unexpected optional dependency import: {name}")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("mupt.builders.all_atom_dpd", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("mupt.builders.all_atom_dpd")

    assert module.AllAtomDPDBuilder.__name__ == "AllAtomDPDBuilder"
    assert module.AllAtomDPDSettings.__name__ == "AllAtomDPDSettings"
    assert module.AllAtomDPDResult.__name__ == "AllAtomDPDResult"
    assert module.AllAtomDPDParameterProvider.__name__ == "AllAtomDPDParameterProvider"


def test_box_length_uses_mass_density_constants():
    from mupt.builders.all_atom_dpd import AMU_TO_G, ANGSTROM3_TO_CM3, AllAtomDPDBuilder, AllAtomDPDSettings

    total_mass_amu = 64000.0
    density_g_cm3 = 2.0
    builder = AllAtomDPDBuilder(settings=AllAtomDPDSettings(density_g_cm3=density_g_cm3))

    expected = ((total_mass_amu * AMU_TO_G / density_g_cm3) / ANGSTROM3_TO_CM3) ** (1.0 / 3.0)

    assert builder._box_length_a(total_mass_amu) == expected


def test_segment_records_counts_tiny_saamr_atoms_and_bonds():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    root, _atoms = _tiny_saamr_hierarchy()

    records = AllAtomDPDBuilder()._segment_records(root)

    assert len(records) == 1
    assert len(records[0].atoms) == 3
    assert records[0].bonds == [(0, 1), (1, 2)]


def test_write_positions_updates_atoms_and_parent_shapes():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    root, atoms = _tiny_saamr_hierarchy()
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
        ]
    )

    AllAtomDPDBuilder._write_positions(root, atoms, positions)

    for atom, position in zip(atoms, positions):
        np.testing.assert_allclose(atom.shape.centroid, position)

    expected_centroid = positions.mean(axis=0)
    segment = root.children[0]
    residue = segment.children[0]
    np.testing.assert_allclose(residue.shape.centroid, expected_centroid)
    np.testing.assert_allclose(segment.shape.centroid, expected_centroid)
    np.testing.assert_allclose(root.shape.centroid, expected_centroid)
