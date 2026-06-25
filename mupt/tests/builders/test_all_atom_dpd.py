"""Fast tests for all-atom DPD builder utilities."""

import builtins
import importlib
import logging
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


def _one_atom_residue_with_connector(label: str, connector_label: str, anchor_position, linker_position):
    atom = Primitive(
        label=f"{label}_C",
        shape=PointCloud(np.array([0.0, 0.0, 0.0])),
        element=elements.C,
        role=PrimitiveRole.PARTICLE,
    )
    connector_handle = atom.register_connector(_chain_connector(anchor_position, linker_position, connector_label))
    residue = Primitive(label=label, role=PrimitiveRole.RESIDUE)
    residue.attach_child(atom)
    return residue, atom, connector_handle


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
        bonds=[(idx, idx + 1) for idx in range(n_residues - 1)],
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


def _tiny_pet_hierarchy() -> tuple[Primitive, dict[str, str]]:
    import networkx as nx

    from mupt.interfaces.rdkit import suppress_rdkit_logs
    from mupt.interfaces.smiles import primitive_from_smiles
    from mupt.mupr.topology import TopologicalStructure

    smiles = {
        "head": "[H]O[CH2][CH2]OC(=O)c1ccc([C:2](=O)-*)cc1",
        "tail": "*-[O:1][CH2][CH2]OC(=O)c1ccc([C:2](=O)O[H])cc1",
    }
    resname_map = {"head": "PTH", "tail": "PTT"}
    lexicon = {}
    with suppress_rdkit_logs():
        for name, residue_smiles in smiles.items():
            residue = primitive_from_smiles(
                residue_smiles,
                ensure_explicit_Hs=True,
                embed_positions=True,
                label=name,
            )
            residue.role = PrimitiveRole.RESIDUE
            residue.metadata["residue_name"] = resname_map[name]
            for atom in residue.children:
                atom.role = PrimitiveRole.PARTICLE
            lexicon[name] = residue

    root = Primitive(label="pet_tiny", role=PrimitiveRole.UNIVERSE)
    segment = Primitive(label="pet_chain_0000", role=PrimitiveRole.SEGMENT)
    handles = []
    for residue_name in ("head", "tail"):
        residue = lexicon[residue_name].copy()
        residue.role = PrimitiveRole.RESIDUE
        for atom in residue.children:
            atom.role = PrimitiveRole.PARTICLE
        handles.append(segment.attach_child(residue))
    segment.set_topology(nx.path_graph(handles, create_using=TopologicalStructure), max_registration_iter=100)
    root.attach_child(segment)
    return root, resname_map


def test_imports_public_symbols_without_hoomd_or_openff(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("hoomd") or name.startswith("openff"):
            raise AssertionError(f"unexpected optional dependency import: {name}")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("mupt.builders.all_atom_dpd", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.import_module("mupt.builders.all_atom_dpd")


def test_box_length_uses_mass_density_constants():
    from mupt.builders.all_atom_dpd import AMU_TO_G, ANGSTROM3_TO_CM3, AllAtomDPDBuilder, AllAtomDPDSettings

    total_mass_amu = 64000.0
    density_g_cm3 = 2.0
    builder = AllAtomDPDBuilder(settings=AllAtomDPDSettings(density_g_cm3=density_g_cm3))

    expected = ((total_mass_amu * AMU_TO_G / density_g_cm3) / ANGSTROM3_TO_CM3) ** (1.0 / 3.0)

    assert builder._box_length_a(total_mass_amu) == expected


def test_explicit_box_lengths_select_orthorhombic_path():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    settings = AllAtomDPDSettings(box_lengths_a=(12.0, 18.0, 24.0), r_cut_a=3.0)
    builder = AllAtomDPDBuilder(settings=settings)

    np.testing.assert_allclose(builder._box_lengths_a(total_mass_amu=1000.0), np.array([12.0, 18.0, 24.0]))
    assert builder._effective_box_length_a(np.array([12.0, 18.0, 24.0])) == pytest.approx(
        (12.0 * 18.0 * 24.0) ** (1.0 / 3.0)
    )


def test_explicit_box_lengths_wrap_orthorhombic_positions():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    np.testing.assert_allclose(
        AllAtomDPDBuilder._wrap(np.array([7.0, 11.0, -14.0]), np.array([10.0, 20.0, 30.0])),
        np.array([-3.0, -9.0, -14.0]),
    )


def test_rejects_nonpositive_density():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    with pytest.raises(ValueError, match="density_g_cm3"):
        AllAtomDPDBuilder(settings=AllAtomDPDSettings(density_g_cm3=0.0))


def test_rejects_too_small_explicit_box_lengths():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    with pytest.raises(ValueError, match="box_lengths_a"):
        AllAtomDPDBuilder(
            settings=AllAtomDPDSettings(box_lengths_a=(12.0, 12.0, 5.0), r_cut_a=3.0)
        )


def test_rejects_invalid_hoomd_device_setting():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    with pytest.raises(ValueError, match="device"):
        AllAtomDPDBuilder(settings=AllAtomDPDSettings(device="TPU"))


@pytest.mark.parametrize(
    "setting,expected",
    [
        ("auto", "auto"),
        ("CPU", "cpu"),
        ("gpu", "gpu"),
    ],
)
def test_hoomd_device_setting_dispatches_requested_device(setting, expected):
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    class FakeDevice:
        @staticmethod
        def auto_select():
            return "auto"

        @staticmethod
        def CPU():
            return "cpu"

        @staticmethod
        def GPU():
            return "gpu"

    class FakeHoomd:
        device = FakeDevice

    builder = AllAtomDPDBuilder(settings=AllAtomDPDSettings(device=setting))

    assert builder._hoomd_device(FakeHoomd) == expected


def test_uniform_chain_length_plan_uses_density_and_explicit_box():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    plan = AllAtomDPDBuilder.plan_uniform_chain_lengths_for_box(
        density_g_cm3=1.0,
        box_lengths_a=(10.0, 20.0, 30.0),
        repeat_unit_mass_amu=50.0,
        chain_length_min=2,
        chain_length_max=5,
        random_seed=123,
    )

    assert plan.box_lengths_a == (10.0, 20.0, 30.0)
    assert all(2 <= length <= 5 for length in plan.chain_lengths)
    assert plan.planned_mass_amu >= plan.target_mass_amu
    assert plan.planned_mass_amu - plan.target_mass_amu < 5 * 50.0


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("r_cut_a", 0.0, "r_cut_a"),
        ("initial_residue_spacing_a", 0.0, "initial_residue_spacing_a"),
        ("initial_angle_max_rad", 0.0, "initial_angle_max_rad"),
        ("initial_angle_max_rad", np.pi + 0.1, "initial_angle_max_rad"),
        ("n_steps_per_interval", 0, "n_steps_per_interval"),
        ("n_steps_max", -1, "n_steps_max"),
        ("report_interval", 0, "report_interval"),
        ("epsilon_reference_mode", "not-a-number", "epsilon_reference_mode"),
        ("nlist_exclusions", ("unsupported",), "nlist_exclusions"),
    ],
)
def test_rejects_invalid_settings(field, value, match):
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    settings = AllAtomDPDSettings()
    setattr(settings, field, value)

    with pytest.raises(ValueError, match=match):
        AllAtomDPDBuilder(settings=settings)


def test_default_settings_use_dense_initialization_restraints():
    from mupt.builders.all_atom_dpd import AllAtomDPDSettings

    settings = AllAtomDPDSettings()

    assert settings.initial_residue_spacing_a == 1.5
    assert settings.bond_scale == 30.0
    assert settings.angle_scale == 30.0
    assert settings.dihedral_scale == 30.0
    assert settings.require_bonded_energy_convergence is True
    assert settings.nlist_exclusions == ("bond",)


def test_nlist_exclusions_are_normalized_and_passed_to_hoomd():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings, _ParameterTables

    captured = {}

    class FakeNList:
        def __init__(self, buffer, exclusions):
            captured["buffer"] = buffer
            captured["exclusions"] = exclusions

    class FakeDPD:
        def __init__(self, nlist, default_r_cut, kT):
            self.params = {}

    class FakeIntegrator:
        def __init__(self, dt):
            self.forces = []
            self.methods = []

    class FakeConstantVolume:
        def __init__(self, filter):
            pass

    class FakeSimulation:
        def __init__(self, device, seed):
            self.operations = type("Operations", (), {"integrator": None, "writers": []})()

        def create_state_from_snapshot(self, frame):
            pass

    class FakeHoomd:
        filter = type("filter", (), {"All": lambda: object()})
        trigger = type("trigger", (), {"Periodic": lambda interval: interval})
        device = type("device", (), {"CPU": staticmethod(lambda: "cpu"), "GPU": staticmethod(lambda: "gpu"), "auto_select": staticmethod(lambda: "auto")})
        Simulation = FakeSimulation
        md = type(
            "md",
            (),
            {
                "Integrator": FakeIntegrator,
                "methods": type("methods", (), {"ConstantVolume": FakeConstantVolume}),
                "nlist": type("nlist", (), {"Cell": FakeNList}),
                "pair": type("pair", (), {"DPD": FakeDPD}),
            },
        )

    class Particles:
        types = ["C"]

    class Frame:
        particles = Particles()

    builder = AllAtomDPDBuilder(
        settings=AllAtomDPDSettings(nlist_exclusions=["bond", "angle"], device="CPU")
    )
    builder._simulation(FakeHoomd, Frame(), [], [], [], [], _ParameterTables(epsilon_by_type={"C": 1.0}))

    assert builder.settings.nlist_exclusions == ("bond", "angle")
    assert captured == {"buffer": 0.4, "exclusions": ("bond", "angle")}


def test_openff_key_atom_indices_support_topology_key_shapes():
    from mupt.builders.all_atom_dpd import OpenFFAllAtomDPDParameterProvider

    class AtomIndicesKey:
        atom_indices = (1, 2, 3)

    class ThisAtomIndexKey:
        this_atom_index = 4

    assert OpenFFAllAtomDPDParameterProvider._atom_indices_from_openff_key(AtomIndicesKey()) == (1, 2, 3)
    assert OpenFFAllAtomDPDParameterProvider._atom_indices_from_openff_key(ThisAtomIndexKey()) == (4,)
    assert OpenFFAllAtomDPDParameterProvider._atom_indices_from_openff_key((5, 6)) == (5, 6)


def test_periodic_idivf_defaults_when_openff_returns_none():
    from mupt.builders.all_atom_dpd import OpenFFAllAtomDPDParameterProvider

    class Parameter:
        idivf = None

    assert OpenFFAllAtomDPDParameterProvider._periodic_idivf(Parameter(), 2) == [1.0, 1.0]


def test_missing_bonded_params_warn_and_use_max_k(caplog):
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    caplog.set_level(logging.WARNING, logger="mupt.builders.all_atom_dpd")

    params = {
        "soft": {"r0": 1.1, "k": 10.0},
        "stiff": {"r0": 1.2, "k": 25.0},
    }

    assigned = AllAtomDPDBuilder._bonded_params_for("missing", params, {"r0": 1.5, "k": 100.0}, "bond")

    assert assigned == {"r0": 1.2, "k": 25.0}
    assert "missing OpenFF bond parameters" in caplog.text
    assert "maximum-k parameter set 'stiff'" in caplog.text


def test_energy_diagnostics_collects_force_energies_per_term():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings, _ParameterTables

    class BondForce:
        energy = 8.0

    class AngleForce:
        energy = 12.0

    class PairForce:
        energy = 20.0

    BondForce.__module__ = "hoomd.md.bond"
    AngleForce.__module__ = "hoomd.md.angle"
    PairForce.__module__ = "hoomd.md.pair"

    class Container:
        def __init__(self, n):
            self.N = n

    class Frame:
        bonds = Container(4)
        angles = Container(6)
        dihedrals = Container(0)
        impropers = Container(0)

    Frame.bonds.types = ["b"]
    Frame.bonds.typeid = np.array([0, 0, 0, 0])
    Frame.angles.types = ["a"]
    Frame.angles.typeid = np.array([0, 0, 0, 0, 0, 0])

    class Integrator:
        forces = [BondForce(), AngleForce(), PairForce()]

    class Operations:
        integrator = Integrator()

    class Simulation:
        operations = Operations()

    parameters = _ParameterTables(
        bond_params={"b": {"k": 100.0, "r0": 1.0}},
        angle_params={"a": {"k": 20.0, "t0": np.pi / 2}},
    )
    settings = AllAtomDPDSettings(bond_energy_tolerance_a=0.2, angle_energy_tolerance_deg=10.0)

    diagnostics = AllAtomDPDBuilder._energy_diagnostics(Simulation(), Frame(), parameters, settings)

    assert diagnostics["counts"] == {"bond": 4, "angle": 6, "dihedral": 0, "improper": 0}
    assert diagnostics["bond_energy"] == 8.0
    assert diagnostics["bond_energy_per_term"] == 2.0
    assert diagnostics["angle_energy"] == 12.0
    assert diagnostics["angle_energy_per_term"] == 2.0
    assert diagnostics["dpd_energy"] == 20.0
    assert diagnostics["dihedral_energy"] is None
    assert diagnostics["dihedral_energy_per_term"] is None
    assert diagnostics["bond_energy_threshold"] == pytest.approx(8.0)
    assert diagnostics["angle_energy_threshold"] == pytest.approx(6 * 0.5 * 20.0 * np.deg2rad(10.0) ** 2)
    assert diagnostics["bond_energy_converged"] is True
    assert diagnostics["angle_energy_converged"] is False
    assert diagnostics["bonded_energy_converged"] is False


def test_convergence_requires_spacing_and_bonded_energy_by_default():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    builder = AllAtomDPDBuilder(settings=AllAtomDPDSettings())

    assert builder._convergence_ok(True, {"bonded_energy_converged": True}) is True
    assert builder._convergence_ok(False, {"bonded_energy_converged": True}) is False
    assert builder._convergence_ok(True, {"bonded_energy_converged": False}) is False

    spacing_only = AllAtomDPDBuilder(
        settings=AllAtomDPDSettings(require_bonded_energy_convergence=False)
    )
    assert spacing_only._convergence_ok(True, {"bonded_energy_converged": False}) is True


@pytest.mark.skipif(importlib.util.find_spec("openff") is None, reason="OpenFF toolkit is not installed")
def test_openff_parameter_provider_handles_pet_improper_idivf_none():
    from mupt.builders.all_atom_dpd import (
        AllAtomDPDBuilder,
        AllAtomDPDSettings,
        OpenFFAllAtomDPDParameterProvider,
    )

    root, resname_map = _tiny_pet_hierarchy()
    provider = OpenFFAllAtomDPDParameterProvider(resname_map=resname_map)
    settings = AllAtomDPDSettings(resname_map=resname_map)
    builder = AllAtomDPDBuilder(settings=settings)

    tables = provider.parameterize(root, builder._segment_records(root), settings)

    assert tables.atom_types_by_global
    assert tables.bond_params
    assert tables.improper_params


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


def test_initial_positions_consumes_placement_generator():
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

    root, _atoms = _tiny_saamr_hierarchy()
    builder = AllAtomDPDBuilder(
        settings=AllAtomDPDSettings(random_seed=123),
        placement_generator=fake_generator,
    )
    records = builder._segment_records(root)

    positions = builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))

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
    builder = AllAtomDPDBuilder(placement_generator=InvalidPlacementGenerator())
    records = builder._segment_records(root)

    with pytest.raises(ValueError, match=match):
        builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))


def test_initial_positions_adapts_nested_residue_layout_to_placement_generator():
    from mupt.builders.base import PlacementGenerator
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    class FakePlacementGenerator(PlacementGenerator):
        def __init__(self):
            self.child_roles = []

        def _generate_placements(self, primitive):
            for idx, (handle, child) in enumerate(primitive.children_by_handle.items()):
                self.child_roles.append(child.role)
                yield handle, RigidTransform.from_translation([10.0 + idx, 0.0, 0.0])

    res1, atom1, res1_right = _one_atom_residue_with_connector(
        "res1", "right", [0.5, 0.0, 0.0], [0.5, 1.0, 0.0]
    )
    res2, atom2, res2_left = _one_atom_residue_with_connector(
        "res2", "left", [-0.5, 0.0, 0.0], [-0.5, -1.0, 0.0]
    )

    group = Primitive(label="group")
    res1_handle = group.attach_child(res1)
    res2_handle = group.attach_child(res2)
    group.connect_children(res1_handle, res1_right, res2_handle, res2_left)
    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    segment.attach_child(group)
    root = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    root.attach_child(segment)

    fake_generator = FakePlacementGenerator()
    builder = AllAtomDPDBuilder(placement_generator=fake_generator)
    records = builder._segment_records(root)

    positions = builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))

    assert fake_generator.child_roles == [PrimitiveRole.RESIDUE, PrimitiveRole.RESIDUE]
    np.testing.assert_allclose(positions, np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]))
    assert records[0].atoms == [atom1, atom2]


def test_initial_positions_preserves_role_order_for_mixed_transparent_layout():
    from mupt.builders.base import PlacementGenerator
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    class LabelPlacementGenerator(PlacementGenerator):
        def __init__(self):
            pass

        def _generate_placements(self, primitive):
            translations = {"res1": 10.0, "res2": 20.0, "res3": 30.0}
            for handle, child in primitive.children_by_handle.items():
                yield handle, RigidTransform.from_translation([translations[child.label], 0.0, 0.0])

    res1, atom1 = _one_atom_residue("res1")
    res2, atom2 = _one_atom_residue("res2")
    res3, atom3 = _one_atom_residue("res3")
    group = Primitive(label="group")
    group.attach_child(res1)
    group.attach_child(res2)
    segment = Primitive(label="seg", role=PrimitiveRole.SEGMENT)
    segment.attach_child(group)
    segment.attach_child(res3)
    root = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    root.attach_child(segment)

    builder = AllAtomDPDBuilder(placement_generator=LabelPlacementGenerator())
    records = builder._segment_records(root)

    positions = builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(123))

    assert records[0].atoms == [atom1, atom2, atom3]
    np.testing.assert_allclose(positions, np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))


def test_default_initial_positions_are_repeatable_for_multi_residue_chain():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    records = [_multi_residue_chain_record()]
    builder = AllAtomDPDBuilder(settings=AllAtomDPDSettings(initial_residue_spacing_a=1.5))

    positions1 = builder._initial_positions(records, box_length=50.0, rng=np.random.default_rng(2468))
    positions2 = builder._initial_positions(records, box_length=50.0, rng=np.random.default_rng(2468))

    np.testing.assert_allclose(positions1, positions2)
    assert positions1.shape == (3, 3)
    assert np.all(positions1 >= -25.0)
    assert np.all(positions1 < 25.0)
    assert np.all(np.linalg.norm(np.diff(positions1, axis=0), axis=1) > 0.0)


def test_default_initial_positions_support_single_residue_segment():
    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder

    root, _atoms = _tiny_saamr_hierarchy()
    builder = AllAtomDPDBuilder()
    records = builder._segment_records(root)
    seed = 123

    positions = builder._initial_positions(records, box_length=100.0, rng=np.random.default_rng(seed))

    target_centroid = np.random.default_rng(seed).uniform(-50.0, 50.0, size=3)
    expected = target_centroid + np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(positions, expected)


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
    builder = AllAtomDPDBuilder(placement_generator=OffsetPlacementGenerator())
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
