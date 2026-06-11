"""Fast tests for all-atom DPD builder utilities."""

import builtins
import importlib
import sys

import numpy as np
import pytest
from periodictable import elements
from scipy.spatial.transform import RigidTransform

from mupt.chemistry.core import BondType
from mupt.geometry.shapes import PointCloud
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


def _chain_connector(anchor_position, linker_position, label: str) -> Connector:
    return Connector(
        anchor=AttachmentPoint({"X"}, position=np.array(anchor_position, dtype=float)),
        linker=AttachmentPoint({"X"}, position=np.array(linker_position, dtype=float)),
        bondtype=BondType.SINGLE,
        label=label,
    )


def _one_atom_residue(label: str) -> tuple[Primitive, Primitive]:
    atom = Primitive(
        label=f"{label}_C",
        shape=PointCloud(np.array([0.0, 0.0, 0.0])),
        element=elements.C,
        role=PrimitiveRole.PARTICLE,
    )
    residue = Primitive(label=label, role=PrimitiveRole.RESIDUE)
    residue.attach_child(atom)
    return residue, atom


def _multi_residue_chain_record(n_residues: int = 3):
    from mupt.builders.all_atom_dpd import _SegmentRecord

    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    residues = []
    atoms = []
    residue_handles = []
    left_connectors = []
    right_connectors = []
    for idx in range(n_residues):
        residue, atom = _one_atom_residue(f"res{idx}")
        left = None
        right = None
        if idx > 0:
            left = residue.register_connector(_chain_connector([-0.5, 0.0, 0.0], [-0.5, -1.0, 0.0], "left"))
        if idx < n_residues - 1:
            right = residue.register_connector(_chain_connector([0.5, 0.0, 0.0], [0.5, 1.0, 0.0], "right"))
        residue_handles.append(segment.attach_child(residue))
        residues.append(residue)
        atoms.append(atom)
        left_connectors.append(left)
        right_connectors.append(right)

    for idx in range(n_residues - 1):
        segment.connect_children(
            residue_handles[idx],
            right_connectors[idx],
            residue_handles[idx + 1],
            left_connectors[idx + 1],
        )

    return _SegmentRecord(
        segment=segment,
        residues=residues,
        atoms=atoms,
        residue_atom_indices=[[idx] for idx in range(n_residues)],
        local_to_global={idx: idx for idx in range(n_residues)},
        bonds=[],
    )


def _tiny_saamr_hierarchy() -> tuple[Primitive, list[Primitive]]:
    """Return universe -> segment -> residue -> H-C-H with two bonds."""

    h1 = Primitive(
        label="H1",
        shape=PointCloud(np.array([0.0, 0.0, 0.0])),
        element=elements.H,
        role=PrimitiveRole.PARTICLE,
    )
    c = Primitive(
        label="C",
        shape=PointCloud(np.array([1.0, 0.0, 0.0])),
        element=elements.C,
        role=PrimitiveRole.PARTICLE,
    )
    h2 = Primitive(
        label="H2",
        shape=PointCloud(np.array([2.0, 0.0, 0.0])),
        element=elements.H,
        role=PrimitiveRole.PARTICLE,
    )

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


def test_rejects_nonpositive_density():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    with pytest.raises(ValueError, match="density_g_cm3"):
        AllAtomDPDBuilder(settings=AllAtomDPDSettings(density_g_cm3=0.0))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("r_cut_a", 0.0, "r_cut_a"),
        ("initial_bond_length_a", 0.0, "initial_bond_length_a"),
        ("initial_angle_max_rad", 0.0, "initial_angle_max_rad"),
        ("initial_angle_max_rad", np.pi + 0.1, "initial_angle_max_rad"),
        ("n_steps_per_interval", 0, "n_steps_per_interval"),
        ("n_steps_max", -1, "n_steps_max"),
        ("report_interval", 0, "report_interval"),
        ("epsilon_reference_mode", "not-a-number", "epsilon_reference_mode"),
    ],
)
def test_rejects_invalid_settings(field, value, match):
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    settings = AllAtomDPDSettings()
    setattr(settings, field, value)

    with pytest.raises(ValueError, match=match):
        AllAtomDPDBuilder(settings=settings)


def test_openff_key_atom_indices_support_topology_key_shapes():
    from mupt.builders.all_atom_dpd import OpenFFAllAtomDPDParameterProvider

    class AtomIndicesKey:
        atom_indices = (1, 2, 3)

    class ThisAtomIndexKey:
        this_atom_index = 4

    assert OpenFFAllAtomDPDParameterProvider._atom_indices_from_openff_key(AtomIndicesKey()) == (1, 2, 3)
    assert OpenFFAllAtomDPDParameterProvider._atom_indices_from_openff_key(ThisAtomIndexKey()) == (4,)
    assert OpenFFAllAtomDPDParameterProvider._atom_indices_from_openff_key((5, 6)) == (5, 6)


def test_build_rejects_malformed_hierarchy_before_optional_imports(monkeypatch):
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "hoomd" or name.startswith("hoomd."):
            raise AssertionError(f"unexpected optional dependency import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    root = Primitive(label="empty", role=PrimitiveRole.UNIVERSE)
    root.attach_child(Primitive(label="orphan_particle", element=elements.C, role=PrimitiveRole.PARTICLE))

    with pytest.raises(ValueError, match="RESIDUE and SEGMENT"):
        AllAtomDPDBuilder().build(root)


def test_segment_records_counts_tiny_saamr_atoms_and_bonds():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    root, _atoms = _tiny_saamr_hierarchy()

    records = AllAtomDPDBuilder()._segment_records(root)

    assert len(records) == 1
    assert len(records[0].atoms) == 3
    assert records[0].bonds == [(0, 1), (1, 2)]


def test_initial_positions_consumes_placement_generator_factory():
    from mupt.builders.base import PlacementGenerator
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    class FakePlacementGenerator(PlacementGenerator):
        def __init__(self) -> None:
            self.seen_labels = []

        def _generate_placements(self, primitive):
            self.seen_labels.append(primitive.label)
            for handle in primitive.children_by_handle:
                yield handle, RigidTransform.from_translation([10.0, 2.0, 3.0])

    fake_generator = FakePlacementGenerator()
    factory_calls = []

    def factory(rng, box_length):
        factory_calls.append(box_length)
        return fake_generator

    root, _atoms = _tiny_saamr_hierarchy()
    builder = AllAtomDPDBuilder(
        settings=AllAtomDPDSettings(random_seed=123),
        placement_generator_factory=factory,
    )
    records = builder._segment_records(root)

    positions = builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))

    assert factory_calls == [100.0]
    assert fake_generator.seen_labels == ["seg"]
    np.testing.assert_allclose(
        positions,
        np.array(
            [
                [10.0, 2.0, 3.0],
                [11.0, 2.0, 3.0],
                [12.0, 2.0, 3.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "mode,match",
    [
        ("missing", "missing="),
        ("duplicate", "duplicates="),
        ("unknown", "unknown="),
    ],
)
def test_initial_positions_validates_placement_generator_handles(mode, match):
    from mupt.builders.base import PlacementGenerator
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    class InvalidPlacementGenerator(PlacementGenerator):
        def __init__(self):
            pass

        def _generate_placements(self, primitive):
            handles = list(primitive.children_by_handle)
            transform = RigidTransform.from_translation([0.0, 0.0, 0.0])
            if mode == "duplicate":
                yield handles[0], transform
                yield handles[0], transform
            elif mode == "unknown":
                yield "unknown", transform

    root, _atoms = _tiny_saamr_hierarchy()
    builder = AllAtomDPDBuilder(placement_generator_factory=lambda rng, box_length: InvalidPlacementGenerator())
    records = builder._segment_records(root)

    with pytest.raises(ValueError, match=match):
        builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))


def test_initial_positions_rejects_nested_residue_layout_for_default_placement():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    residue, _atom = _one_atom_residue("res")
    group = Primitive(label="group")
    group.attach_child(residue)
    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    segment.attach_child(group)
    root = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    root.attach_child(segment)

    builder = AllAtomDPDBuilder()
    records = builder._segment_records(root)

    with pytest.raises(ValueError, match="immediate child.*RESIDUE-role"):
        builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))


def test_default_initial_positions_are_repeatable_for_multi_residue_chain():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    records = [_multi_residue_chain_record()]
    builder = AllAtomDPDBuilder(settings=AllAtomDPDSettings(initial_bond_length_a=1.5))

    positions1 = builder._initial_positions(records, box_length=50.0, rng=np.random.default_rng(2468))
    positions2 = builder._initial_positions(records, box_length=50.0, rng=np.random.default_rng(2468))

    np.testing.assert_allclose(positions1, positions2)
    assert positions1.shape == (3, 3)
    assert np.all(positions1 >= -25.0)
    assert np.all(positions1 < 25.0)
    assert np.all(np.linalg.norm(np.diff(positions1, axis=0), axis=1) > 0.0)


def test_default_initial_positions_do_not_mutate_global_numpy_rng():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    records = [_multi_residue_chain_record()]
    builder = AllAtomDPDBuilder()

    np.random.seed(13579)
    expected = np.random.random(4)
    np.random.seed(13579)

    builder._initial_positions(records, box_length=50.0, rng=np.random.default_rng(1234))

    np.testing.assert_allclose(np.random.random(4), expected)


def test_initial_positions_wraps_atoms_for_periodic_snapshot():
    from mupt.builders.base import PlacementGenerator
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    class OffsetPlacementGenerator(PlacementGenerator):
        def __init__(self):
            pass

        def _generate_placements(self, primitive):
            for handle in primitive.children_by_handle:
                yield handle, RigidTransform.from_translation([8.0, 0.0, 0.0])

    root, _atoms = _tiny_saamr_hierarchy()
    builder = AllAtomDPDBuilder(placement_generator_factory=lambda rng, box_length: OffsetPlacementGenerator())
    records = builder._segment_records(root)

    positions = builder._initial_positions(records, box_length=10.0, rng=np.random.default_rng(123))

    np.testing.assert_allclose(
        positions,
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


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
