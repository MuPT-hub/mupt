"""
Strategies for checking and enacting spatial anti-alignment of pairs of
Connectors which comprise a connection. Models bonding in 3D space
"""

from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from scipy.spatial.transform import RigidTransform

if TYPE_CHECKING:
    from .connectors import Connector


class ConnectorAntialignmentStrategy(ABC):
    """
    Defines interface for antialigning one Connector with another
    I.e. Transforming one connector so that its linker is coincident to
    the other's anchor and vice versa WITHOUT modifying the `to_connector`
    """

    @abstractmethod
    def antialignment_transformation(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> RigidTransform:
        """
        A rigid transformation applied to `align_connector` to line it up
        with `to_connector` for the antialignment procedure implemented here
        """
        ...

    @abstractmethod
    def _antialign(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
    ) -> None:
        """
        Implementation of how Connector `align_connector` should be
        acted on to antialign it to Connector `to_connector`

        Note: do NOT include changes to bond length here;
        those are bundled automatically with `antialign()`
        """
        # DEV: made this a separate method (rather than always
        # just applying `self.antialignment_transformation()`)
        # to allow alignment techniques potentially not based
        # on rigid transformations to fit within this framework
        ...

    def antialign(
        self,
        align_connector: "Connector",
        to_connector: "Connector",
        match_bond_length: bool = False,
        dihedral_angle_rad: Optional[float] = None,
    ) -> None:
        """
        Apply the requisite antialignment transformation and other modifications
        to `align_connector`so that it is antialigned with `to_connector` in-place
        WITHOUT modifying `to_connector` itself

        If match_bond_length = True, will also stretch/compress bond
        length on `align_connector` to match length of `to_connector`
        """
        self._antialign(
            align_connector=align_connector,
            to_connector=to_connector,
        )
        if match_bond_length:
            align_connector.set_bond_length(to_connector.bond_length)

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
        Return a copy of `align_connector` which is antialigned with `to_connector`
        WITHOUT modifying either `align_connector` or `to_connector`

        Non-in-place version of `self.antialign()`
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
        `align_connector` along the mutual bond axis to that angle
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
