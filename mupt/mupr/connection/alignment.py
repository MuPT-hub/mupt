"""
Strategies for checking and enacting spatial anti-alignment of pairs of
Connectors which comprise a connection. Models bonding in 3D space
"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import Iterator, Optional
from abc import ABC, abstractmethod
from contextlib import contextmanager

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

    # Dihedral angles
    ## Preconditions
    def _precondition_dihedral_antialign(
        self,
        align_connector: Connector,
        to_connector: Connector,
        alignment_tolerance: float = 1e-6,
    ) -> None:
        """
        Precondition to ensure necessary conditions are met for
        Connector dihedral operations are well-defined, viz:
        * Both Connectors have a dihedral plane defined
        * The two Connectors are mutually antialigned
        """
        if not (
            align_connector.has_dihedral_orientation
            and to_connector.has_dihedral_orientation
        ):
            raise ValueError(
                "Cannot compute dihedral alignment between Connectors "
                "without explicitly-defined dihedral plane orientations"
            )

        # essential bond vectors be antialigned for dihedral assignment to make sense
        # any dihedral angle assignment should only move the subspace orthogonal to
        # the span of the two bond vectors, i.e. the dihedral rotation transformations
        # have eigenvectors parallel to the bond vector(s) with eigenvalue 1.0
        if not align_connector.is_antialigned(to_connector, within=alignment_tolerance):
            raise ValueError(
                "Cannot modify dihedral angle between non-antialigned Connectors"
            )

    ## Calculation of rigid transformations
    def dihedral_taring_transform(
        self,
        align_connector: Connector,
        to_connector: Connector,
        alignment_tolerance: float = 1e-6,
        suppress_precondition: bool = False,
    ) -> RigidTransform:
        """
        Transformation which rotates align_connector about its bond vector so its
        normal vector is parallel to that of to_connector, assuming both Connectors
        are mutually antiantialigned a priori
        """
        if not suppress_precondition:
            self._precondition_dihedral_antialign(
                align_connector, to_connector, alignment_tolerance=alignment_tolerance
            )

        # don't need to normalize, already done by alignment rotation calculation
        normal_alignment_rotation: Rotation = alignment_rotation(
            align_connector.normal_vector,
            to_connector.normal_vector,
        )

        return align_connector.rotation_about_anchor(normal_alignment_rotation)

    def dihedral_increment_transform(
        self,
        align_connector: Connector,
        to_connector: Connector,
        dihedral_angle_increment_rad: float = 0.0,
        alignment_tolerance: float = 1e-6,
        suppress_precondition: bool = False,
    ) -> RigidTransform:
        """
        Transformation which rotates align_connector about its bond vector so the
        angle between the Connectors' dihedral planes changes by the prescribed amount.
        Angle is measured in radians (by default, assigned 0.0) and is taken
        FROM the normal of to_connector TO the normal of align_connector
        """
        if not suppress_precondition:
            self._precondition_dihedral_antialign(
                align_connector, to_connector, alignment_tolerance=alignment_tolerance
            )

        # DEV: purposely NOT using align_connector-only check, since we REQUIRE
        # antialignment for dihedral assignment to be valid (stronger precondition)
        dihedral_increment_rotation: Rotation = Rotation.from_rotvec(
            dihedral_angle_increment_rad * -to_connector.unit_bond_vector
            # dihedral_angle_increment_rad * align_connector.unit_bond_vector
        )

        return align_connector.rotation_about_anchor(dihedral_increment_rotation)

    def dihedral_assignment_transform(
        self,
        align_connector: Connector,
        to_connector: Connector,
        dihedral_angle_rad: float = 0.0,
        alignment_tolerance: float = 1e-6,
    ) -> RigidTransform:
        """
        Transformation which rotates align_connector about its bond vector so the
        angle between the dihedral planes of the Connectors subtends the chosen angle.
        Angle is measured in radians (by default, assigned 0.0) and is taken
        FROM the normal of to_connector TO the normal of align_connector
        """
        # DEV: ALWAYS validate dihedral preconditions, suppress on sub-operations
        self._precondition_dihedral_antialign(
            align_connector, to_connector, alignment_tolerance=alignment_tolerance
        )

        dihedral_tare_transform: RigidTransform = self.dihedral_taring_transform(
            align_connector,
            to_connector,
            alignment_tolerance=alignment_tolerance,
            suppress_precondition=True,  # checked just prior
        )
        dihedral_increment_transform: RigidTransform = (
            self.dihedral_increment_transform(
                align_connector,
                to_connector,
                dihedral_angle_rad,
                alignment_tolerance=alignment_tolerance,
                suppress_precondition=True,  # checked just prior
            )
        )

        return dihedral_increment_transform * dihedral_tare_transform

    ## Application of dihedral rotations
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

    def increment_dihedral(
        self,
        align_connector: Connector,
        to_connector: Connector,
        dihedral_angle_increment_rad: float = 0.0,
        alignment_tolerance: float = 1e-6,
    ) -> None:
        """
        Set the dihedral angle between this Connector and
        another Connector by rigidly transforming this Connector
        """
        LOGGER.info(
            f"incrementing dihedral angle between Connectors {align_connector.label} "
            f"and {to_connector.label} to {dihedral_angle_increment_rad} radians"
        )
        align_connector.rigidly_transform(
            transformation=self.dihedral_increment_transform(
                align_connector=align_connector,
                to_connector=to_connector,
                dihedral_angle_increment_rad=dihedral_angle_increment_rad,
                alignment_tolerance=alignment_tolerance,
            )
        )

    # Antialignment
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
        with self._antialignment_one_sided(
            align_connector,
            to_connector=to_connector,
            roll_back_transform=roll_back_transform,
            match_bond_length=match_bond_length,
        ) as transformation:
            if roll_back_transform:
                return transformation

    @contextmanager
    def _antialignment_one_sided(
        self,
        align_connector: Connector,
        to_connector: Connector,
        *,
        roll_back_transform: bool,
        match_bond_length: bool,
    ) -> Iterator[RigidTransform]:
        """Apply an alignment and optionally undo its rigid motion on exit."""
        transformation = self.antialignment_transformation(
            align_connector, to_connector=to_connector
        )
        align_connector.rigidly_transform(transformation)
        try:
            if match_bond_length:
                self._match_bond_length(
                    align_connector=align_connector,
                    to_connector=to_connector,
                )
            yield transformation
        finally:
            if roll_back_transform:
                align_connector.rigidly_transform(transformation.inv())

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
        alignment_tolerance: float = 1e-6,
    ) -> Optional[tuple[RigidTransform, RigidTransform]]:
        """
        Antialign each connector in turn, accommodating asymmetric strategies.

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
            Whether to undo both rigid transformations after the pair has been
            aligned and return them in connector order. Bond length adjustments
            are retained.

            If True, will suppress dihedral angle assignment, EVEN if an angle is given
        dihedral_angle_rad: Optional[float], default None
            An optional dihedral angle in radians which, if supplied, sets the
            dihedral angle between the pair of Connectors

            Dihedral angle is defined as the angle betwwen Connector normal
            vectors IFF the two bond vectors are antialigned

            If not supplied, performs no dihedral rotation
        alignment_tolerance : float, default 1e-6
            The maximum anchor-to-linker distance error to allow
            and still consider the two Connectors antialigned

            Only needed if dihedral transformation is applied

        Returns
        -------
        antialignment_transformations: Optional[tuple[RigidTransform, RigidTransform]]
            If roll_back_transform=True, returns that rigid transformations that
            WOULD HAVE BEEN applied to align_connector and to_connector, in that order

            Useful for transforming attached bodies & avoiding a local double-transform
        """
        # TB: may be more robust to do the "reverse" antalignment first (i.e. match
        # to_connector to align_connector). Skipping to avoid downstream breaks for now
        with (
            self._antialignment_one_sided(
                align_connector,
                to_connector=to_connector,
                match_bond_length=True,
                roll_back_transform=roll_back_transform,
            ) as align_connector_transform,
            self._antialignment_one_sided(
                to_connector,
                to_connector=align_connector,
                match_bond_length=True,
                roll_back_transform=roll_back_transform,
            ) as to_connector_transform,
        ):
            if dihedral_angle_rad is not None:
                if not roll_back_transform:
                    self.assign_dihedral(
                        align_connector,
                        to_connector,
                        dihedral_angle_rad=dihedral_angle_rad,
                        alignment_tolerance=alignment_tolerance,
                    )
                # if rolling back, append dihedral transform to
                # alignment transform without actually applying it
                else:
                    dihedral_transform: RigidTransform = (
                        self.dihedral_assignment_transform(
                            align_connector=align_connector,
                            to_connector=to_connector,
                            dihedral_angle_rad=dihedral_angle_rad,
                            alignment_tolerance=alignment_tolerance,
                        )
                    )

                    align_connector_transform = (
                        dihedral_transform * align_connector_transform
                    )

        if roll_back_transform:
            return align_connector_transform, to_connector_transform

    # TODO: also add antialigned_mutually? TB - unclear if/when this is necessary


class ConnectorAntialignmentRigid(ConnectorAntialignmentStrategy):
    """
    Antialignment strategy which works purely through rigid motions, i.e.
    only translates and rotates align_connector without distorting or modifying it
    """

    def antialignment_transformation(
        self,
        align_connector: Connector,
        to_connector: Connector,
    ) -> RigidTransform:
        """
        Compute a rigid transformation which:
        1) translates align_connector so its anchor coincides with to_connector's linker
        2) rotates align_connector about its anchor so its bond vector is antiparallel
           to the span of to_connector's bond vector

        If the two Connectors have the same bond length, the anchor of align_connector
        will be coincident with the linker of the other; otherwise, the anchor will
        merely lay on the span of the to_connectors bond vector
        """
        bond_antialignment: Rotation = alignment_rotation(
            align_connector.unit_bond_vector, -to_connector.unit_bond_vector
        )

        # order of application reads bottom-to-top (rightmost operator acts first)
        return (
            RigidTransform.from_translation(to_connector.linker.position)
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
        antialign_rotation: Rotation = alignment_rotation(
            align_connector.bond_vector,
            to_connector.anchor.position - align_connector.anchor.position,
        )

        return align_connector.rotation_about_anchor(antialign_rotation)

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
