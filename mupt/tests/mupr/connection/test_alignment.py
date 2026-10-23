"""Unit tests for Connector bond antialignment protocols"""

# ruff: noqa: F401

import pytest
from typing import Hashable, Optional

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.geometry.coordinates.directions import random_unit_vector
from mupt.mupr.connection.connectors import Connector, AttachmentPoint
from mupt.mupr.connection.alignment import (
    are_antialigned,
    ConnectorAntialignmentStrategy,
    ConnectorAntialignmentRigid,
    ConnectorAntialignmentBallistic,
)


def randomly_placed_connector(
    anchor_marker: Hashable = 0,
    linker_marker: Hashable = 1,
    L: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> Connector:
    """
    Produce a marked Connector with both attachment point
    positions chosen uniformly randomly from [-L, L]**3
    """
    sampler = np.random if rng is None else rng
    return Connector(
        anchor=AttachmentPoint(
            {anchor_marker},
            attachment=anchor_marker,
            position=sampler.uniform(low=-L, high=L, size=3),
        ),
        linker=AttachmentPoint(
            {linker_marker},
            attachment=linker_marker,
            position=sampler.uniform(low=-L, high=L, size=3),
        ),
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

    assert are_antialigned(conn, conn_counter, within=tolerance) == expected_antialigned


@pytest.mark.parametrize(
    "alignment",
    [
        ConnectorAntialignmentRigid(),
        # ConnectorAntialignmentBallistic(),
    ],
)
class TestAntialignment:
    """
    Tests for the calculation and application methods each
    ConnectorAntialignmentStrategy is expected to implement
    """

    def test_antialignment_application(
        self, alignment: ConnectorAntialignmentStrategy
    ) -> None:
        """Test that"""
        conn_a = randomly_placed_connector(0, 1)
        conn_b = randomly_placed_connector(1, 0)

        alignment.antialign(conn_b, to_connector=conn_a, match_bond_length=True)

        assert are_antialigned(conn_a, conn_b)
