"""
Strategies for checking and enacting spatial anti-alignment of pairs of
Connectors which comprise a connection. Models bonding in 3D space
"""

from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from numpy.linalg import norm
from scipy.spatial.transform import Rotation, RigidTransform
from ...geometry.transforms.rigid.rotations import alignment_rotation

from ...geometry.measure import compare_optional_positions

if TYPE_CHECKING:
    from .connectors import Connector


def are_antialigned(
    align_connector: "Connector",
    to_connector: "Connector",
    within: float = 1e-6,
) -> bool:
    """
    Whether align_connector is anti-aligned with to_connector, i.e. whether
    the anchor of align_connector is within some cutoff distance of the linker
    of the to_connector, and vice-versa (with the same tolerance for both)

    This check is commutative, i.e. are_antialigned(Ca, Cb) = are_antialigned(Cb, Ca)

    Parameters
    ----------
    align_connector: Connector
        The Connector to transform during antialignment
    to_connector: Connector
        The Connector being antialigned to
        This Connector is NOT transformed and remains static
    within: float, default 1E-6
        Absolute tolerance for distance between both Connectors's positions
        Can be thought of as the radius of bounding spheres around attachment points

    Returns
    -------
    are_antialigned: bool
        Whether the pair of Connectors are antialigned within the chosen tolerance
    """
    return compare_optional_positions(
        align_connector.anchor.position,
        to_connector.linker.position,
        radius=within,
    ) and compare_optional_positions(
        align_connector.linker.position,
        to_connector.anchor.position,
        radius=within,
    )


class ConnectorAntialignmentStrategy(ABC):
    """
    Defines interface for antialigning one Connector with another
    I.e. Transforming one connector so that its linker is coincident to
    the other's anchor and vice versa WITHOUT modifying the to_connector
    """

    @abstractmethod
    def antialignment_transformation(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> RigidTransform:
        """
        A rigid transformation applied to align_connector to line it up
        with to_connector for the antialignment procedure implemented here
        """
        ...

    @abstractmethod
    def _match_bond_length(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> None:
        """
        Ensure both connectors have the same bond length
        (i.e. anchor-to-linker distance) after an alignment
        transformation has been applied
        """
        ...

    def antialign(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
        match_bond_length: bool = False,
        dihedral_angle_rad: Optional[float] = None,
    ) -> None:
        """
        Apply one-half of a complete, mutual Connector antialignment by manipulating
        align_connector to antialign at to_connector WITHOUT modifying to_connector.
        As such, to_connector is not guaranteed to be antialigned after this process.

        One-sided antialignment proceeds in a 3-step fashion:
        1. Apply a rigid transformation to bring align_connector into antialignment
        2. Match bond lengths between the two Connectors
        3. Assign a stipulated dihedral angle

        All 3 of these steps are optional and can be disabled to accomodate various
        downstream coordinate alignment workflows; care is required when deciding
        when and how a callershould do so

        Parameters
        ----------
        align_connector: Connector
            The Connector being manipulated by this antialignment transform
        to_connector: Connector
            The Connector which align_connector is being antialigned to
            This Connector will NOT be adulterated in any way during this
            process,regardless of which (if any) of the 3 steps are performed
        match_bond_length: bool, default False
            Flag which determines whether to match the bond
            length of align_connector with that of to_connector

            Bond length is defined here as the anchor-to-linker vector norm
        dihedral_angle_rad: Optional[float], default None
            An optional dihedral angle in radians which, if supplied, sets the
            dihedral angle between the pair of Connectors

            Dihedral angle is defined as the angle betwwen Connector normal
            vectors IFF the two bond vectors are antialigned

            If not supplied, performs no dihedral rotation
        """
        transformation: RigidTransform = self.antialignment_transformation(
            align_connector,
            to_connector=to_connector,
        )
        align_connector.rigidly_transform(transformation=transformation)

        if match_bond_length:
            self._match_bond_length(
                align_connector=align_connector,
                to_connector=to_connector,
            )

        # NOTE: sentinel (rather than default 0.0) weakens
        # preconditions on tangents when no dihedral is specified
        if dihedral_angle_rad is not None:
            align_connector.assign_dihedral(
                to_connector,
                dihedral_angle_rad=dihedral_angle_rad,
            )

    def antialigned(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
        match_bond_length: bool = False,
        dihedral_angle_rad: Optional[float] = None,
    ) -> "Connector":
        """
        Return a copy of align_connector which is antialigned with to_connector
        WITHOUT modifying either align_connector or to_connector

        Non in-place version of self.antialign()
        """
        align_connector_new = align_connector.copy()
        self.antialign(
            align_connector_new,
            to_connector=to_connector,
            match_bond_length=match_bond_length,
            dihedral_angle_rad=dihedral_angle_rad,
        )
        return align_connector_new

    def mutually_antialign(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
        dihedral_angle_rad: Optional[float] = None,
    ) -> None:
        """
        Apply anti-alignment implemented here first to
        Designed to accomodate assymetric alignment schemes

        If a dihedral angle is provided, will also rotate
        align_connector along the mutual bond axis to that angle
        """
        self.antialign(
            align_connector,
            to_connector=to_connector,
            match_bond_length=True,
            dihedral_angle_rad=None,  # dihedrals done at end
        )
        self.antialign(
            to_connector,
            to_connector=align_connector,
            match_bond_length=True,
            dihedral_angle_rad=None,  # dihedrals done at end
        )
        # DEV: asymmetry relative to rigid alignment viz dihedral angles is no accident;
        # Rigid alignment results in antialignment after one application with bond
        # length matching, whereas ballistic alignment in general requires both
        # Connectors to be mutually transformed to guarantee antialignment

        # NOTE: sentinel (rather than default 0.0) weakens
        # preconditions on tangents when no dihedral is specified
        if dihedral_angle_rad is not None:
            align_connector.assign_dihedral(
                to_connector,
                dihedral_angle_rad=dihedral_angle_rad,
            )


class ConnectorAntialignmentRigid(ConnectorAntialignmentStrategy):
    """
    Antialignment strategy which works purely through rigid motions, i.e.
    only translates and rotates align_connector without distorting or modifying it
    """

    def __init__(self, tare_dihedrals: bool = False) -> None:
        self.tare_dihedrals = tare_dihedrals

    def antialignment_transformation(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> RigidTransform:
        """
        Compute a rigid transformation which antialigns a pair of
        Connectors by making the linker point of align_connector
        coincident with the anchor of to_connector

        If the two Connectors have the same bond length, the anchor of align_connector
        will be coincident with the linker of the other; otherwise, the anchor will
        merely lay on the span of the to_connectors bond vector

        If tare_dihedrals is True (default False), will also ensure
        that the dihedral planes of the two Connectors are coplanar.
        This may be desirable in many cases, but comes with stricter
        preconditions, namely both connectors having tangents define
        """
        bond_antialignment: Rotation = alignment_rotation(
            align_connector.unit_bond_vector, -to_connector.unit_bond_vector
        )

        if self.tare_dihedrals:
            tangent_alignment = alignment_rotation(
                bond_antialignment.apply(align_connector.tangent_vector),
                to_connector.tangent_vector,
            )
        else:
            tangent_alignment = Rotation.identity()

        # order of application reads bottom-to-top (rightmost operator acts first)
        return (
            RigidTransform.from_translation(to_connector.linker.position)
            * RigidTransform.from_rotation(tangent_alignment)
            * RigidTransform.from_rotation(bond_antialignment)
            * RigidTransform.from_translation(-align_connector.anchor.position)
        )

    def _match_bond_length(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> None:
        """Set bond length of align_connector to that of to_connector"""
        align_connector.set_bond_length(to_connector.bond_length)


class ConnectorAntialignmentBallistic(ConnectorAntialignmentStrategy):
    """
    Antialignment strategy which points-and-aims at to-connector
    without requiring any rigid motion of adjoining bodies

    Called "ballistic" because the action (especially when matching bond length)
    resembles align_connector aiming and then "shooting" its linker at to_connector
    """

    def antialignment_transformation(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> RigidTransform:
        """
        Compute a rigid transformation which aligns a pair of Connectors by turning
        the bond vector of align_connector to face the linker point of to_connector
        The anchor positions of either Connector will be unaffected
        """
        return (
            RigidTransform.from_translation(align_connector.anchor.position)
            * RigidTransform.from_rotation(
                alignment_rotation(
                    align_connector.bond_vector,
                    to_connector.anchor.position - align_connector.anchor.position,
                ),
            )
            * RigidTransform.from_translation(-align_connector.anchor.position)
        )

    def _match_bond_length(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> None:
        """
        Set align_connector's bond length to be the distance
        from its anchor to the anchor of to_connector
        """
        separation = norm(
            align_connector.anchor.position - to_connector.anchor.position
        )
        align_connector.set_bond_length(separation)
