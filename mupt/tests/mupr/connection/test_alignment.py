"""Unit tests for Connector bond antialignment protocols"""

import pytest
from itertools import product as cartesian

import numpy as np
from numpy.testing import assert_allclose

from . import randomly_placed_connector
from mupt.geometry.measure import compare_optional_positions, normalized
from mupt.geometry.coordinates.directions import random_unit_vector, random_tangent

from mupt.mupr.connection.connectors import AttachmentPoint, Connector
from mupt.mupr.connection.exceptions import ConnectorAntialignmentError
from mupt.mupr.connection.alignment import (
    ConnectorAntialignmentStrategy,
    ConnectorAntialignmentRigid,
    ConnectorAntialignmentBallistic,
)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random number shared among tests here"""
    return np.random.default_rng(seed=5772)


## Tests proper
@pytest.mark.parametrize(
    "perturbation,tolerance,expected_antialigned",
    [
        (0.0, 1e-6, True),
        (0.0, 1e-12, True),
        (1e-6, 1e-12, False),
        (1e-6, 1e-5, True),
        (1e-12, 1e-6, True),
        (1e-1, 1e0, True),
        (1e-2, 1e2, True),
        (1e2, 1e-2, False),
        # DEV: comparison exactly ON the surface of an
        # epsilon ball is dicey; result is essentially random
        (1e-6 - 1e-8, 1e-6 + 1e-8, True),
        (1e-6 + 1e-8, 1e-6 - 1e-8, False),
    ],
)
def test_are_antialigned(
    perturbation: float,
    tolerance: float,
    expected_antialigned: bool,
    rng: np.random.Generator,
) -> None:
    """
    Test that antialignment check correctly identifies or
    rejects antialigned Connectors to within tolerance
    """
    conn = randomly_placed_connector(0, 1)
    conn_counter = conn.counterpart()
    conn_counter.anchor.position += perturbation * random_unit_vector(rng=rng)

    assert conn.is_antialigned(conn_counter, within=tolerance) == expected_antialigned


@pytest.mark.parametrize(
    "alignment",
    [
        ConnectorAntialignmentRigid(),
        ConnectorAntialignmentBallistic(),
    ],
)
class TestAntialignment:
    """
    Tests for the calculation and application methods each
    ConnectorAntialignmentStrategy is expected to implement
    """

    def test_one_sided_alignment_no_bond_match(
        self, alignment: ConnectorAntialignmentStrategy
    ) -> None:
        """
        Test that one-sided antialignment WITHOUT bond length matching
        causes the bond_vector of align_connector to point from its anchor
        to the anchor of to_connector (but not necessarily the other way around)

        More precisely, check the align_connectors bond vector lies on
        the span of the vector difference between the two anchor sites
        """
        conn_a = randomly_placed_connector(0, 1)
        conn_b = randomly_placed_connector(1, 0)

        alignment.antialign_one_sided(
            conn_b, to_connector=conn_a, match_bond_length=False
        )

        similarity: float = np.dot(
            conn_b.unit_bond_vector,
            # order of subtraction guarantees same direction, precludes -1.0
            normalized(conn_a.anchor.position - conn_b.anchor.position),
        )

        assert_allclose(similarity, 1.0)

    def test_one_sided_antialignment_bond_match(
        self, alignment: ConnectorAntialignmentStrategy
    ) -> None:
        """
        Test that one-sided antialignment WITH bond matching causes the bond
        vector of align_connector to face the anchor of to_connector AND the
        linker of align_connector to be coincident with the anchor of to_connector
        """
        conn_a = randomly_placed_connector(0, 1)
        conn_b = randomly_placed_connector(1, 0)

        alignment.antialign_one_sided(
            conn_b, to_connector=conn_a, match_bond_length=True
        )

        assert compare_optional_positions(
            conn_b.linker.position,
            conn_a.anchor.position,
        )

    def test_mutual_antialignment(
        self, alignment: ConnectorAntialignmentStrategy
    ) -> None:
        """
        Test that mutual antialignment does indeed result in
        both Connectors being antialigned to one another
        """
        conn_a = randomly_placed_connector(0, 1)
        conn_b = randomly_placed_connector(1, 0)

        alignment.antialign_mutually(conn_b, to_connector=conn_a)

        assert conn_a.is_antialigned(conn_b)

    def test_mutual_antialignment_with_rollback(
        self, alignment: ConnectorAntialignmentStrategy
    ) -> None:
        """Replaying the returned transforms reproduces direct mutual alignment."""
        conn_a = Connector(
            AttachmentPoint(position=np.array([0.0, 0.0, 0.0])),
            AttachmentPoint(position=np.array([0.0, 0.0, 1.0])),
        )
        conn_b = Connector(
            AttachmentPoint(position=np.array([3.0, 0.0, 0.0])),
            AttachmentPoint(position=np.array([3.0, 2.0, 0.0])),
        )
        direct_a, direct_b = conn_a.copy(), conn_b.copy()
        alignment.antialign_mutually(direct_a, direct_b)
        assert direct_a.is_antialigned(direct_b)

        transforms = alignment.antialign_mutually(
            conn_a, conn_b, roll_back_transform=True
        )
        assert transforms is not None
        assert_allclose(conn_a.anchor.position, [0.0, 0.0, 0.0], atol=1e-8)
        assert_allclose(conn_b.anchor.position, [3.0, 0.0, 0.0], atol=1e-8)
        assert_allclose(conn_a.unit_bond_vector, [0.0, 0.0, 1.0], atol=1e-8)
        assert_allclose(conn_b.unit_bond_vector, [0.0, 1.0, 0.0], atol=1e-8)

        for conn, direct, transform in zip(
            (conn_a, conn_b), (direct_a, direct_b), transforms, strict=True
        ):
            assert_allclose(conn.bond_length, direct.bond_length)
            conn.rigidly_transform(transform)
            assert_allclose(conn.anchor.position, direct.anchor.position, atol=1e-8)
            assert_allclose(conn.linker.position, direct.linker.position, atol=1e-8)


# Dihedral transformations
def example_dihedral_angles() -> tuple[float, ...]:
    """Collection of deihdral angles for testing various parts of precision range"""
    return (
        0.0,
        1e-12,
        1e-6,
        2 * np.pi / 1_000,
        np.pi / 6,
        np.pi / 4,
        1.0,
        np.pi / 3,
        np.pi / 2,
        np.pi,
        np.pi + 1e-6,
        np.pi - 1e-6,
        2.0,
        2 / 3 * np.pi,
        3 / 2 * np.pi,
        2 * np.pi - 1e-6,
    )


@pytest.mark.parametrize(
    "antialign_strategy",
    [ConnectorAntialignmentRigid(), ConnectorAntialignmentBallistic()],
)
def test_dihedral_guard_missing_dihedral_planes(
    antialign_strategy: ConnectorAntialignmentStrategy,
) -> None:
    """
    Test that the dihedral precondition method catches pairs of
    Connectors in which one of both do not have a defined tangent
    vector (and by extension, dihedral plane orientation)
    """
    conn_a = Connector()
    conn_b = Connector()

    with pytest.raises(
        ConnectorAntialignmentError,
        match="undefined dihedral planes",
    ):
        antialign_strategy._precondition_dihedral_antialign(conn_a, conn_b)


@pytest.mark.parametrize(
    "antialign_strategy",
    [ConnectorAntialignmentRigid(), ConnectorAntialignmentBallistic()],
)
def test_dihedral_guard_non_antialigned(
    antialign_strategy: ConnectorAntialignmentStrategy,
) -> None:
    """
    Test that the dihedral precondition method catches pairs of Connectors
    which have not been antialigned (i.e. anchors and linkers don't overlap)
    """
    conn_a = Connector(
        anchor=AttachmentPoint({0}, position=np.array([0.0, 0.0, 0.0])),
        linker=AttachmentPoint({1}, position=np.array([0.0, 2.0, 0.0])),
    )  # need to have valid tangents defined, but bond vectors non-antialigned
    conn_a.tangent_vector = np.array([0.0, 0.0, 3.0])

    conn_b = Connector(
        anchor=AttachmentPoint({1}, position=np.array([1.0, 0.0, 0.0])),
        linker=AttachmentPoint({0}, position=np.array([1.0, 0.0, 1.0])),
    )  # need to have valid tangents defined, but bond vectors non-antialigned
    conn_b.tangent_vector = np.array([-3.0, 0.0, 0.0])

    with pytest.raises(
        ConnectorAntialignmentError,
        match="non-antialigned Connectors",
    ):
        antialign_strategy._precondition_dihedral_antialign(conn_a, conn_b)


@pytest.mark.parametrize(
    "dihedral_angle_rad,antialign_strategy",
    cartesian(
        example_dihedral_angles(),
        [ConnectorAntialignmentRigid(), ConnectorAntialignmentBallistic()],
    ),
)
def test_dihedral_assignment(
    dihedral_angle_rad: float,
    antialign_strategy: ConnectorAntialignmentStrategy,
) -> None:
    """
    Test that antialigned Connectors can have their dihedral angles
    configured to any desired angle about the mutual bond vector span
    """
    conn_a = randomly_placed_connector(assign_tangent=True)
    conn_b = randomly_placed_connector(assign_tangent=True)

    antialign_strategy.antialign_mutually(
        conn_b,
        to_connector=conn_a,
        dihedral_angle_rad=dihedral_angle_rad,
    )
    assert conn_b.is_antialigned(conn_a)

    assert_allclose(
        np.cos(dihedral_angle_rad),
        np.dot(conn_a.unit_normal_vector, conn_b.unit_normal_vector),
        atol=1e-12,  # DEV: this epsilon may need adjustment
    )


@pytest.mark.parametrize(
    "dihedral_angle_increment_rad,antialign_strategy",
    cartesian(
        example_dihedral_angles(),
        [ConnectorAntialignmentRigid(), ConnectorAntialignmentBallistic()],
    ),
)
def test_increment_dihedral(
    dihedral_angle_increment_rad: float,
    antialign_strategy: ConnectorAntialignmentStrategy,
) -> None:
    """
    Test that antialigned Connectors can have their dihedral angles
    incremented by a desired amount while remaining antialigned
    """
    conn_a: Connector = randomly_placed_connector(assign_tangent=True)
    conn_b: Connector = conn_a.counterpart()
    conn_b.tangent_vector = random_tangent(conn_b.bond_vector)

    conn_b_normal_init = conn_b.unit_normal_vector[:]
    antialign_strategy.increment_dihedral(
        align_connector=conn_b,
        to_connector=conn_a,
        dihedral_angle_increment_rad=dihedral_angle_increment_rad,
    )
    conn_b_normal_final = conn_b.unit_normal_vector[:]

    assert conn_a.is_antialigned(conn_b)
    assert_allclose(
        np.dot(conn_b_normal_final, conn_b_normal_init),
        np.cos(dihedral_angle_increment_rad),
        atol=1e-12,
    )
