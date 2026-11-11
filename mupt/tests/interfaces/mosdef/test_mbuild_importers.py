"""Tests for reading mBuild Compound positions back onto a MuPT hierarchy"""

import numpy as np
import pytest

pytest.importorskip("mbuild")

from anytree.iterators import PreOrderIter

from mupt.geometry.shapes import PointCloud
from mupt.interfaces.mosdef.mbuild import (
    primitive_to_mbuild,
    update_primitive_from_mbuild,
)
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.mupr.connection.alignment import ConnectorAntialignmentRigid
from mupt.mupr.primitives import CompositePrimitive, RootPrimitive
from mupt.mupr.primitives.predicates import is_atom


def _build_chain(n_units: int) -> CompositePrimitive:
    """Short polyethylene chain from embedded repeat units, placed then linked"""
    head = primitive_from_smiles(
        "[H:1]-[C:1](-[H])(-[H])-[C:2](-[H])(-[H])-[*:2]",
        label="A",
        embed_positions=True,
    )
    mid = primitive_from_smiles(
        "[*:1]-[C:1](-[H])(-[H])-[C:2](-[H])(-[H])-[*:2]",
        label="B",
        embed_positions=True,
    )
    tail = primitive_from_smiles(
        "[*:1]-[C:1](-[H])(-[H])-[C:2](-[H])(-[H])-[H:2]",
        label="C",
        embed_positions=True,
    )
    chain = CompositePrimitive(label="chain", role="segment")
    units = [head.copy()] + [mid.copy() for _ in range(n_units - 2)] + [tail.copy()]
    for unit in units:
        unit.role = "residue"
        chain.attach_child(unit)
    align = ConnectorAntialignmentRigid()
    for a, b in zip(units, units[1:]):
        x, y = next(
            (x, y)
            for x in a.connections.connectors_free
            for y in b.connections.connectors_free
            if x.bondable_with(y)
        )
        x.set_bond_length(1.54)
        y.set_bond_length(1.54)
        b.rigidly_transform(
            align.antialignment_transformation(align_connector=y, to_connector=x)
        )
        a.connect_neighbor(b, our_connector=x, their_connector=y)
    return chain


@pytest.fixture(scope="function")
def system() -> RootPrimitive:
    """Two linked 4-unit PE chains with coordinates (Angstrom)"""
    return RootPrimitive(
        label="system", role="universe", children=[_build_chain(4), _build_chain(4)]
    )


def _atoms(prim) -> list:
    return [node for node in PreOrderIter(prim) if is_atom(node)]


def _bonded_connectors(prim) -> list:
    return [
        conn
        for atom in _atoms(prim)
        for conn in atom.connections.connectors
        if conn.has_neighbor
    ]


def _assert_parent_pointclouds_match_atoms(prim) -> None:
    for unit in PreOrderIter(prim):
        if is_atom(unit) or not isinstance(unit.shape, PointCloud):
            continue
        atom_points = np.vstack([
            np.asarray(a.shape.positions).reshape(-1, 3) for a in _atoms(unit)
        ])
        assert np.allclose(
            np.sort(np.asarray(unit.shape.positions).reshape(-1, 3), axis=0),
            np.sort(atom_points, axis=0),
        )


def test_unchanged_compound_leaves_positions_unchanged(system: RootPrimitive) -> None:
    """Reading back an untouched Compound does not move any atom"""
    before = {a.address: np.asarray(a.shape.centroid).copy() for a in _atoms(system)}
    compound = primitive_to_mbuild(
        system, predicate=is_atom, preserve_intermediates=True, coords_to_nm=0.1
    )
    update_primitive_from_mbuild(system, compound)
    for atom in _atoms(system):
        assert np.allclose(atom.shape.centroid, before[atom.address], atol=1e-6)


@pytest.mark.parametrize("preserve_intermediates", [True, False])
def test_positions_written_by_address(
    system: RootPrimitive, preserve_intermediates: bool
) -> None:
    """Every atom lands on the position of the mBuild particle carrying its address"""
    compound = primitive_to_mbuild(
        system,
        predicate=is_atom,
        preserve_intermediates=preserve_intermediates,
        coords_to_nm=0.1,
    )
    rng = np.random.default_rng(0)
    compound.xyz = compound.xyz + rng.normal(scale=0.05, size=compound.xyz.shape)

    moved = update_primitive_from_mbuild(system, compound, nm_to_coords=10.0)

    assert len(moved) == compound.n_particles
    by_address = {a.address: a for a in _atoms(system)}
    for particle in compound.particles():
        assert np.allclose(
            by_address[particle.address].shape.centroid, particle.pos * 10.0
        )


def test_connectors_follow_atoms(system: RootPrimitive) -> None:
    """Bonded Connectors stay on their atoms and antialigned with their neighbors"""
    compound = primitive_to_mbuild(
        system, predicate=is_atom, preserve_intermediates=True, coords_to_nm=0.1
    )
    rng = np.random.default_rng(1)
    compound.xyz = compound.xyz + rng.normal(scale=0.05, size=compound.xyz.shape)
    update_primitive_from_mbuild(system, compound)

    bonded = _bonded_connectors(system)
    assert bonded
    for conn in bonded:
        assert np.allclose(conn.anchor.position, conn.holder.shape.centroid)
        assert conn.is_antialigned(conn.neighbor)


def test_parent_shapes_refreshed(system: RootPrimitive) -> None:
    """Repeat-unit PointClouds are rebuilt from their atoms after an update"""
    compound = primitive_to_mbuild(
        system, predicate=is_atom, preserve_intermediates=True, coords_to_nm=0.1
    )
    compound.translate([1.0, -2.0, 0.5])  # nm
    update_primitive_from_mbuild(system, compound)
    _assert_parent_pointclouds_match_atoms(system)


def test_coarse_grained_particles_move_rigidly(system: RootPrimitive) -> None:
    """Repeat units exported as beads: atoms under each bead move with it rigidly"""
    is_unit = lambda prim: getattr(prim, "role", None) == "residue"  # noqa: E731
    units = [node for node in PreOrderIter(system) if is_unit(node)]
    internal_before = {
        u.address: np.asarray([a.shape.centroid for a in _atoms(u)])
        - np.asarray(u.shape.centroid)
        for u in units
    }
    compound = primitive_to_mbuild(
        system, predicate=is_unit, preserve_intermediates=True, coords_to_nm=0.1
    )
    compound.translate([0.3, 0.3, 0.3])
    update_primitive_from_mbuild(system, compound)

    for unit in units:
        internal_after = np.asarray([
            a.shape.centroid for a in _atoms(unit)
        ]) - np.asarray(unit.shape.centroid)
        assert np.allclose(internal_after, internal_before[unit.address], atol=1e-6)
    for particle in compound.particles():
        unit = next(u for u in units if u.address == particle.address)
        assert np.allclose(unit.shape.centroid, particle.pos * 10.0)
    for conn in _bonded_connectors(system):
        assert conn.is_antialigned(conn.neighbor)


def test_nm_to_coords_scaling(system: RootPrimitive) -> None:
    """nm_to_coords sets the conversion from mBuild nanometers to hierarchy units"""
    compound = primitive_to_mbuild(
        system, predicate=is_atom, preserve_intermediates=True, coords_to_nm=1.0
    )
    compound.translate([0.5, 0.0, 0.0])
    update_primitive_from_mbuild(system, compound, nm_to_coords=1.0)
    by_address = {a.address: a for a in _atoms(system)}
    for particle in compound.particles():
        assert np.allclose(by_address[particle.address].shape.centroid, particle.pos)


def test_particle_without_address_rejected(system: RootPrimitive) -> None:
    """A Compound not made by primitive_to_mbuild is refused"""
    import mbuild as mb

    compound = mb.Compound(name="foreign")
    compound.add(mb.Particle(name="C", pos=[0.0, 0.0, 0.0], element="C"))
    with pytest.raises(ValueError, match="no MuPT address"):
        update_primitive_from_mbuild(system, compound)


def test_address_from_other_hierarchy_rejected(system: RootPrimitive) -> None:
    """Particles exported from a different hierarchy are refused"""
    other = RootPrimitive(label="other", children=[_build_chain(3)])
    compound = primitive_to_mbuild(other, predicate=is_atom, coords_to_nm=0.1)
    with pytest.raises(ValueError, match="not in the hierarchy"):
        update_primitive_from_mbuild(system, compound)


def test_non_finite_position_rejected(system: RootPrimitive) -> None:
    """NaN positions are refused instead of being written onto the hierarchy"""
    compound = primitive_to_mbuild(
        system, predicate=is_atom, preserve_intermediates=True, coords_to_nm=0.1
    )
    first = next(iter(compound.particles()))
    first.pos = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        update_primitive_from_mbuild(system, compound)
