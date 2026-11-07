"""
Strategies for checking and enacting spatial anti-alignment of pairs of
Connectors which comprise a connection. Models bonding in 3D space
"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import Optional
from abc import ABC, abstractmethod

from numpy.linalg import norm
from scipy.spatial.transform import Rotation, RigidTransform

from .connectors import Connector
from ...geometry.transforms.rigid.rotations import alignment_rotation


class ConnectorAntialignmentStrategy(ABC):
    """
    Defines interface for antialigning one Connector with another
    I.e. Transforming one connector so that its linker is coincident to
    the other's anchor and vice versa WITHOUT modifying the to_connector
    """

    def dihedral_assignment_transform(
        self,
        align_connector: Connector,
        to_connector: Connector,
        dihedral_angle_rad: float = 0.0,
        alignment_tolerance: float = 1e-6,
    ) -> RigidTransform:
        """
        Transformation which, when applied to this Connector, rotates it
        so that the dihedral planes between the Connectors subtends the
        desired dihedral angle in radians (by default, 0.0 rad)

        It is required (and enforced) for the pair of Connectors
        to be antialigned for this operation to be valid
        """
        if not (
            align_connector.has_dihedral_orientation
            and to_connector.has_dihedral_orientation
        ):
            raise ValueError(
                "Cannot compute dihedral alignment between Connectors "
                "without explicitly-defined dihedral plane orientations"
            )

        # DEV: could technically weaken to check when bond vectors are antiparallel
        # (i.e. -1 dot product when normed) and difference between anchors is parallel
        # and antiparallel with bond vectors respectively, but didn't for simplicity
        if not align_connector.is_antialigned(to_connector, within=alignment_tolerance):
            raise ValueError(
                "Cannot set dihedral angle with non-antialigned Connectors"
            )

        # recall, tangent vectors are invariant under translation
        tangent_alignment = alignment_rotation(
            align_connector.tangent_vector,
            to_connector.tangent_vector,
        )
        # BUG: extra degree of freedom here; need to ensure tangents also line up
        dihedral_rotation = Rotation.from_rotvec(
            dihedral_angle_rad * to_connector.unit_bond_vector
        )
        # align tangents before setting dihedral to avoid inter-tangent angle calc
        dihedral_alignment = dihedral_rotation * tangent_alignment

        return (
            RigidTransform.from_translation(align_connector.anchor.position)
            * RigidTransform.from_rotation(dihedral_alignment)
            * RigidTransform.from_translation(-align_connector.anchor.position)
        )

    def assign_dihedral(
        self,
        align_connector: Connector,
        to_connector: Connector,
        dihedral_angle_rad: float = 0.0,
        alignment_tolerance: float = 1e-6,
    ) -> None:
        """
        Set the dihedral angle between this Connector and
        another Connector by rigidly transforming this Connector
        """
        LOGGER.info(
            f"Setting dihedral angle between Connectors {align_connector.label} "
            f"and {to_connector.label} to {dihedral_angle_rad} radians"
        )
        align_connector.rigidly_transform(
            transformation=self.dihedral_assignment_transform(
                align_connector=align_connector,
                to_connector=to_connector,
                dihedral_angle_rad=dihedral_angle_rad,
                alignment_tolerance=alignment_tolerance,
            )
        )

    @abstractmethod
    def antialignment_transformation(
        self,
        align_connector: Connector,
        to_connector: Connector,
    ) -> RigidTransform:
        """
        A rigid transformation applied to align_connector to line it up
        with to_connector for the antialignment procedure implemented here
        """
        ...

    @abstractmethod
    def _match_bond_length(
        self,
        align_connector: Connector,
        to_connector: Connector,
    ) -> None:
        """
        Ensure both connectors have the same bond length
        (i.e. anchor-to-linker distance) after an alignment
        transformation has been applied
        """
        ...

    def antialign_one_sided(
        self,
        align_connector: Connector,
        to_connector: Connector,
        roll_back_transform: bool = False,
        match_bond_length: bool = False,
    ) -> Optional[RigidTransform]:
        """
        Apply one-half of a complete, mutual Connector antialignment by manipulating
        align_connector to antialign at to_connector WITHOUT modifying to_connector.
        As such, to_connector is not guaranteed to be antialigned after this process.

        One-sided antialignment proceeds in a 2-step fashion:
        1. Apply a rigid transformation to bring align_connector into antialignment
        2. Match bond lengths between the two Connectors

        Any of these 2 steps are optional and can be disabled to accomodate various
        downstream coordinate alignment workflows; care is required when deciding
        when and how a caller should do so

        Parameters
        ----------
        align_connector: Connector
            The Connector being manipulated by this antialignment transform
        to_connector: Connector
            The Connector which align_connector is being antialigned to
            This Connector will NOT be adulterated in any way during this
            process,regardless of which (if any) of the 3 steps are performed
        roll_back_transform: bool, default False
            Flag which dictates whether to undo the rigid transformation
            applied to align_connector after bond lengths have been matched

            If True, this method will return the (non-inverted) transformation
            If False, will return None instead

            The use-case for this setting is aligning owning structures using
            Connector antialignments, i.e. to hand back the rigid transformation
            needed to move attached non-Connector parts into place while also
            making sure to not double-apply it to this Connector
        match_bond_length: bool, default False
            Flag which dictates whether to match the bond
            length of align_connector with that of to_connector

            Bond length is defined here as the anchor-to-linker vector norm

        Returns
        -------
        antialignment_transformation: Optional[RigidTransform]
            If roll_back_transform=True, returns that rigid transformation that
            WOULD HAVE BEEN applied in the first step; otherwise return None

            Useful for transforming attached bodies & avoiding a local double-transform
        """
        antialign_transformation: RigidTransform = self.antialignment_transformation(
            align_connector,
            to_connector=to_connector,
        )
        align_connector.rigidly_transform(transformation=antialign_transformation)

        if match_bond_length:
            self._match_bond_length(
                align_connector=align_connector,
                to_connector=to_connector,
            )

        if roll_back_transform:
            # TB TODO: this is kind of the most wasteful approach, but was
            # the quickest to get working on the timeline for beta release;
            # revisit this and make more efficient
            align_connector.rigidly_transform(antialign_transformation.inv())
            return antialign_transformation

    def antialigned_one_sided(
        self,
        align_connector: Connector,
        to_connector: Connector,
        match_bond_length: bool = False,
    ) -> Connector:
        """
        Return a copy of align_connector which is antialigned with to_connector
        WITHOUT modifying either align_connector or to_connector

        Non in-place version of self.antialign_one_sided()
        """
        align_connector_new = align_connector.copy()
        self.antialign_one_sided(
            align_connector_new,
            to_connector=to_connector,
            roll_back_transform=False,  # copy sidesteps worry about moving original
            match_bond_length=match_bond_length,
        )
        return align_connector_new

    def antialign_mutually(
        self,
        align_connector: Connector,
        to_connector: Connector,
        roll_back_transform: bool = False,
        dihedral_angle_rad: Optional[float] = None,
    ) -> Optional[tuple[RigidTransform, RigidTransform]]:
        """
        Apply anti-alignment implemented here first to
        Designed to accomodate assymetric alignment schemes

        If a dihedral angle is provided, will also rotate
        align_connector along the mutual bond axis to that angle

        Parameters
        ----------
        align_connector: Connector
            Connector which is first one-sidedly antialigned to the second
        to_connector: Connector
            Connector which is initially alaigned to, and which is then antialigned
            one-sidedly to the first Connector after the first is antialigned
        roll_back_transform: bool=False
            Whether to revert the rigid transformations that would be applied to
            either Connector and return those transformation, in Connecto order
            See ConnectorAntialignmentStrategy.antialign_one_sided() for more details

            If True, will suppress dihedral angle assignment, EVEN if an angle is given
        dihedral_angle_rad: Optional[float], default None
            An optional dihedral angle in radians which, if supplied, sets the
            dihedral angle between the pair of Connectors

            Dihedral angle is defined as the angle betwwen Connector normal
            vectors IFF the two bond vectors are antialigned

            If not supplied, performs no dihedral rotation

        Returns
        -------
        antialignment_transformations: Optional[tuple[RigidTransform, RigidTransform]]
            If roll_back_transform=True, returns that rigid transformations that
            WOULD HAVE BEEN applied to align_connector and to_connector, in that order

            Useful for transforming attached bodies & avoiding a local double-transform
        """
        # N.B.: note that the specifics of the transforms we support here
        # are what enable running rollbacks sequentially without fear
        # In general this is not true, and should be examined if adding new alignments
        # * For rigid antalignment, the second transformation must be the identity
        #   hence the transform obtained from swapping the Connectors would just
        #   be the inverse transform of the first here
        # * For ballistic antialignment, by construction nothing attached to the
        # anchor moves; only a rotation about the anchor point is performed
        align_connector_transform: Optional[RigidTransform] = self.antialign_one_sided(
            align_connector,
            to_connector=to_connector,
            match_bond_length=True,
            roll_back_transform=roll_back_transform,
        )
        to_connector_transform: Optional[RigidTransform] = self.antialign_one_sided(
            to_connector,
            to_connector=align_connector,
            match_bond_length=True,
            roll_back_transform=roll_back_transform,
        )

        if roll_back_transform:
            if dihedral_angle_rad is not None:
                LOGGER.warning(
                    "Overriding dihedral angle assignment, since the "
                    "prerequisite rigid transformations were rolled back"
                )
            return (align_connector_transform, to_connector_transform)  # type: ignore

        # NOTE: sentinel (rather than default 0.0) weakens
        # preconditions on tangents when no dihedral is specified
        if dihedral_angle_rad is not None:
            self.assign_dihedral(
                align_connector,
                to_connector,
                dihedral_angle_rad=dihedral_angle_rad,
            )

    # TODO: also add antialigned_mutually? TB - unclear if/when this is necessary


class ConnectorAntialignmentRigid(ConnectorAntialignmentStrategy):
    """
    Antialignment strategy which works purely through rigid motions, i.e.
    only translates and rotates align_connector without distorting or modifying it
    """

    def __init__(self, tare_dihedrals: bool = False) -> None:
        self.tare_dihedrals = tare_dihedrals

    def antialignment_transformation(
        self,
        align_connector: Connector,
        to_connector: Connector,
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
        align_connector: Connector,
        to_connector: Connector,
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
        align_connector: Connector,
        to_connector: Connector,
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
        align_connector: Connector,
        to_connector: Connector,
    ) -> None:
        """
        Set align_connector's bond length to be the distance
        from its anchor to the anchor of to_connector
        """
        separation = norm(
            align_connector.anchor.position - to_connector.anchor.position
        )
        align_connector.set_bond_length(separation)
