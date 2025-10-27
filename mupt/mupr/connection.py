'''Abstractions of connections between two primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Hashable,
    Iterable,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)
from warnings import warn

from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
from itertools import product as cartesian

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from ..chemistry.core import BondType
from ..geometry.arraytypes import Shape, Vector3, as_n_vector, compare_optional_positions
from ..geometry.measure import within_ball
from ..geometry.coordinates.basis import is_orthonormal
from ..geometry.transforms.linear import rejector
from ..geometry.transforms.rigid.rotations import alignment_rotation
from ..geometry.transforms.rigid.application import RigidlyTransformable


# Label typehints
ConnectorLabel = TypeVar('ConnectorLabel', bound=Hashable)
ConnectorHandle = tuple[ConnectorLabel, int]
AttachmentLabel = TypeVar('AttachmentLabel', bound=Hashable) # TODO: narrow down this type as use cases become clearer

# Custom Exceptions
class ConnectionError(Exception):
    '''Raised when Connector-related errors as encountered'''
    pass

class IncompatibleConnectorError(ConnectionError):
    '''Raised when attempting to connect two Connectors which are, for whatever reason, incompatible'''
    pass

class MissingConnectorError(ConnectionError):
    '''Raised when a required Connector is missing'''
    pass

class UnboundConnectorError(ConnectionError):
    '''Raised when a pair of Connectors are unexpectedly not bound to one another'''
    pass

# Helper classes
class TraversalDirection(Enum):
    '''
    Uniquifying label indicating whether a connection faces "forward" or "backward" along a path graph 
    relative to an arbitrary-but-consistent absolute direction of traversal along the path from end-to-end
    '''
    AMBI = 0
    ANTERO = 1
    RETRO = 2
    
    @classmethod
    def complement(cls, direction : 'TraversalDirection') -> 'TraversalDirection':
        '''
        Get the complement (i.e. "opposite") direction to a given TraversalDirection
        
        Parameters
        ----------
        direction : TraversalDirection
            The direction to get the complement of
            
        Returns
        -------
        TraversalDirection
            The complement of the given direction
        '''
        if direction == cls.ANTERO:
            return cls.RETRO
        elif direction == cls.RETRO:
            return cls.ANTERO
        elif direction == cls.AMBI:
            return cls.AMBI

# DEV: would love to make this frozen, but that breaks the RigidlyTansformable mechanism under-the-hood,
# and also prevents reassignment of the attachment label, which is important in some cases
@dataclass(frozen=False)
class AttachmentPoint(RigidlyTransformable):
    '''
    A point with an associated attachment, which must come from a predefined set (attachables) of allowable designations.
    Forms half of a Connector; represents a spatial attachment to some other body, identified by its attachment.
    '''
    attachables : set[AttachmentLabel] = field(default_factory=set)
    attachment : Optional[AttachmentLabel] = field(default=None)
    position : np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    
    def __setattr__(self, key, value):
        if key == 'attachment':
            if (value is not None) and (value not in self.attachables):
                raise ValueError(f'Attachment "{value!s}" not designated as one of attachable labels {self.attachables}')
        if key == 'position':
            value = as_n_vector(value, 3)
        return super().__setattr__(key, value)
        
    # Implementing RigidTransformable contracts
    def _copy_untransformed(self) -> 'AttachmentPoint':
        return self.__class__(
            attachables=set(att for att in self.attachables),
            attachment=self.attachment,
            position=np.array(self.position, copy=True)
        )
        
    def _rigidly_transform(self, transformation: RigidTransform) -> None:
        self.position[:] = transformation.apply(self.position)

# Connector class proper
class Connector(RigidlyTransformable):
    '''Abstraction of the notion of a chemical bond between a known body (anchor) and an indeterminate neighbor body (linker)'''
    DEFAULT_LABEL : ClassVar[ConnectorLabel] = 'Conn'
    
    def __init__(
        self,
        anchor : Optional[AttachmentPoint]=None,
        linker : Optional[AttachmentPoint]=None,
        bondtype : BondType=BondType.UNSPECIFIED,
        query_smarts : str='',
        label : Optional[ConnectorLabel]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ):
        self.anchor = anchor if (anchor is not None) else AttachmentPoint()
        self.linker = linker if (linker is not None) else AttachmentPoint()
        
        self.bondtype = bondtype
        self.query_smarts = query_smarts
        self.label = self.__class__.DEFAULT_LABEL if (label is None) else label
        self.metadata = metadata or dict()
    
        self._tangent_position = None # DEV: no call to setter; must be assigned via protected tangent_vector property

    # Geometric properties
    ## DEV: implemented vector properties (e.g. bond/tangent/normal) by tracking endpoint positions under the hood to get them to
    ## preserving relative orientations for local orthogonal basis under general rigid transformations; key observation is that 
    ## a DIFFERENCE between positions is invariant under shifts of the origin, i.e. if v = (a - b), Tv = T(a - b) = T(a) - T(b), 
    
    ## Attachment site position wrappers - DEV: necessary for backward compatibility with attr reference, though could be deprecated eventually
    @property
    def anchor_position(self) -> Vector3:
        '''The central position that this Connector is anchored to'''
        warn('Connector.anchor_position is slated for deprecation; use Connector.anchor.position instead', category=DeprecationWarning)
        return self.anchor.position
        
    @property
    def linker_position(self) -> Vector3:
        '''The position of the off-body linker point'''
        warn('Connector.linker_position is slated for deprecatation; use Connector.linker.position instead', category=DeprecationWarning)
        return self.linker.position

    ## Bond vector
    @property
    def has_bond_vector(self) -> bool:
        '''Determine whether this Connector has a bond vector (i.e. definite spanning direction away from anchor) defined'''
        # TODO: determine appropriate practical tolerances for "nonzero" bond vector
        return not np.allclose(self.anchor.position, self.linker.position, rtol=1E-6, atol=1E-8) 
    
    @property
    def bond_vector(self) -> Vector3:
        '''A vector spanning from the anchor position to the position of the off-body linker'''
        if not self.has_bond_vector:
            raise FloatingPointError('Anchor and linker positions of Connector as nearer than preset tolerance and cannot be distinguished to determine bond vector')
        return self.linker.position - self.anchor.position
    
    @bond_vector.setter
    def bond_vector(self, new_bond_vector : Vector3) -> None:
        # TODO: cast this as a rigid transformation of linker to track cumulative transform? (would enable reset of bond length history)
        self.linker.position = as_n_vector(new_bond_vector, 3) + self.anchor.position
        
    @property
    def bond_length(self) -> float:
        '''Distance spanned by the bond vector - i.e. distance from anchor to linker positions'''
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> Vector3:
        '''Unit vector in the same direction as the bond (oriented from anchor to linker)'''
        return self.bond_vector / self.bond_length # DEV: use normalized()?
    
    def set_bond_length(self, new_bond_length : float) -> None:
        '''Adjust length of bond vector by moving linker position along the bond vector's span, keeping the anchor fixed in place'''
        self.bond_vector = new_bond_length * self.unit_bond_vector

    ## Tangent vector
    @property
    def has_tangent_position(self) -> bool:
        '''Determine whether this Connector has a tangent position (i.e. point defining dihedral orientation) defined'''
        return self._tangent_position is not None
       
    @property
    def tangent_vector(self) -> Vector3:
        '''
        Vector tangent to the dihedral plane and orthogonal to the bond vector
        
        The tangent and bond vectors span the dihedral plane and 
        fix a local right-handed coordinate system for the Connector
        '''
        if not self.has_tangent_position:
            raise AttributeError('Tangent position of Connector unassigned')
        return self._tangent_position - self.anchor.position
        
    @tangent_vector.setter
    def tangent_vector(self, new_tangent_vector : Vector3) -> None:
        '''Update tangent positions given a new tangent vector'''
        new_tangent_vector = as_n_vector(new_tangent_vector, 3)
        if not np.isclose(
            np.dot( # DEV: opting not to normalize here in case either vector has small magnitude - revisit if that becomes an issue
                self.bond_vector,
                new_tangent_vector,
            ),
            0.0
        ):
            raise ValueError('Badly-set tangent vector is not orthogonal to the bond vector of the Connector')
        
        self._tangent_position = new_tangent_vector + self.anchor.position # DEV: move validation of tangent position orthogonality into here?
        
    @property
    def unit_tangent_vector(self) -> Vector3:
        '''Unit vector in the same direction as the tangent vector'''
        return self.tangent_vector / np.linalg.norm(self.tangent_vector)

    def set_tangent_from_coplanar_point(self, coplanar_point : Vector3) -> None:
        '''Set point tangent to the dihedral plane and orthogonal to the linker point from any third point in the dihedral plane'''
        self.tangent_vector = rejector(self.bond_vector) @ (coplanar_point - self.anchor.position)

    def set_tangent_from_normal_point(self, normal_point : Vector3) -> None:
        '''Set point tangent to the dihedral plane and orthogonal to the linker point from a point on the span of the normal to the dihedral plane'''
        self.tangent_vector = np.cross(self.bond_vector, normal_point - self.anchor.position)
        
    ## Normal vector
    @property
    def normal_vector(self) -> Vector3:
        '''A vector normal to the dihedral plane and orthogonal to both the bond and tangent vectors'''
        return np.cross(self.bond_vector, self.tangent_vector)
    
    def unit_normal_vector(self) -> Vector3:
        '''Unit vector in the same direction as the normal vector'''
        return self.normal_vector / np.linalg.norm(self.normal_vector)
    
    ## Local orthonormal basis (formed from unit bond, tangent, and normal vectors)
    @property
    def has_dihedral_orientation(self) -> bool:
        '''Determine whether this Connector has a dihedral orientation (i.e. tangent position) defined'''
        return self.has_bond_vector and self.has_tangent_position
    has_local_orthogonal_basis = has_dihedral_orientation # alias
    
    def local_orthonormal_basis(self) -> np.ndarray[Shape[Literal[3, 3]], float]:
        '''
        Return a 3x3 array representing an orthonormal basis for this Connector's local coordinate system
        Columns of the array are the basis vectors, which are all mutually orthogonal and of unit length
        
        Basis vectors are in fact the unit bond, tangent, and normal vectors associated to this Connector, respectively
        '''
        local_orthonormal_basis = np.vstack([
            self.unit_bond_vector,
            self.unit_tangent_vector,
            self.unit_normal_vector,
        ]).T # DEV: transpose to get basis vectors as columns
        if not is_orthonormal(local_orthonormal_basis):
            raise ValueError('Bond, tangent, and normal vectors of Connector are not mutually orthonormal')
        
        return local_orthonormal_basis
    
    # Applying rigid transformations (fulfilling RigidlyTransformable contracts)
    def _copy_untransformed(self) -> 'Connector':
        new_connector = self.__class__(
            anchor=self.anchor.copy(),
            linker=self.linker.copy(),
            bondtype=self.bondtype,
            query_smarts=str(self.query_smarts),
            label=self._label,
            metadata=deepcopy(self.metadata),
        )
        if self.has_tangent_position:
            new_connector.tangent_vector = as_n_vector(self.tangent_vector, 3)

        return new_connector

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.anchor.rigidly_transform(transformation)
        self.linker.rigidly_transform(transformation)
        if self.has_tangent_position:
            self._tangent_position = transformation.apply(self._tangent_position)

    # Anti-aligning Connectors to one another (simulates bonding in 3D space)
    ## DEV: eventually try to move as much of the implementation of these transforms to geometry.transforms.rigid as possible
    def are_antialigned(self, other : 'Connector', within : float=1E-6) -> bool:
        ## DEV: was unsure of whether or not to make this a classmethod; opted for instance method instead, with the understanding
        ## that you can still call it like a classmethod (i.e. conn1.align(conn2) <-> Connector.align(conn1, conn2))
        '''
        Whether this Connector is anti-aligned with another Connector, i.e. whether 
        the anchor of this Connector is within some cutoff distance of the linker
        of the other Connector, and vice-versa (with the same tolerance for both)
        '''
        return (
            within_ball(
                self.anchor.position,
                other.linker.position,
                radius=within,
            )
            and within_ball(
                self.linker.position,
                other.anchor.position,
                radius=within,
            )
        )
        
    ## Dihedral angle
    def dihedral_assignment_transform(
        self,
        other : 'Connector',
        dihedral_angle_rad : float=0.0,
        alignment_tolerance : float=1E-6,
    ) -> RigidTransform:
        '''
        Transformation which, when applied to this Connector, rotates it so that the dihedral planes
        between the Connectors subtends the desired dihedral angle in radians (by default, 0.0 rad)
        It is required (and enforced) for the pair of Connectors to be antialigned for this operation to be valid
        '''
        if not (self.has_dihedral_orientation and other.has_dihedral_orientation):
            raise ValueError('Cannot compute dihedral alignment between Connectors without explicitly-defined dihedral plane orientations')
        
        if not self.are_antialigned(other, within=alignment_tolerance):
            # DEV: could technically weaken this check to when bond vectors are antiparallel (-1 dot product when normed)
            # and difference between anchors is parallel and antiparallel with bond vectors respectively, but didn't for simplicity
            raise ValueError('Cannot set dihedral angle with non-antialigned Connectors')
        
        tangent_alignment = alignment_rotation(self.tangent_vector, other.tangent_vector) # recall, these are invariant under translation
        # dihedral_rotation = Rotation.from_rotvec(-dihedral_angle_rad * self.unit_bond_vector) # minus accounts for reversed direction; positive with other works equally well
        dihedral_rotation = Rotation.from_rotvec(dihedral_angle_rad * other.unit_bond_vector)
        dihedral_alignment = dihedral_rotation * tangent_alignment # first align tangents, then set dihedral (avoids explicit inter-tangent angle calculation)

        return (
            RigidTransform.from_translation(self.anchor.position)
            * RigidTransform.from_rotation(dihedral_alignment)
            * RigidTransform.from_translation(-self.anchor.position)
        )

    def assign_dihedral(
        self,
        other : 'Connector',
        dihedral_angle_rad : float=0.0,
        alignment_tolerance : float=1E-6,
    ) -> None:
        '''Set the dihedral angle between this Connector and another Connector by rigidly transforming this Connector'''
        LOGGER.info(f'Setting dihedral angle between Connectors {self.label} and {other.label} to {dihedral_angle_rad} rad')
        self.rigidly_transform(
            transformation=self.dihedral_assignment_transform(
                other,
                dihedral_angle_rad=dihedral_angle_rad,
                alignment_tolerance=alignment_tolerance
            )
        )
        
    def with_assigned_dihedral(
        self,
        other : 'Connector',
        dihedral_angle_rad : float=0.0,
        alignment_tolerance : float=1E-6,
    ) -> 'Connector':
        '''Return a copy of this Connector with the dihedral angle set relative to another Connector'''
        new_connector = self.copy()
        new_connector.assign_dihedral(
            other,
            dihedral_angle_rad=dihedral_angle_rad,
            alignment_tolerance=alignment_tolerance
        )
        
        return new_connector

    ## Rigid alignment
    def rigid_antialignment_to(
        self, 
        other : 'Connector',
        tare_dihedrals : bool=False,
    ) -> RigidTransform:
        '''
        Compute a rigid transformation which antialigns a pair of Connectors by making
        the linker point of this Connector coincident with the anchor of the other Connector
        
        If the two Connectors have the same bond length, the anchor of this Connector will be coincident with the linker
        of the other; otherwise, the anchor will merely lay on the span of the other Connectors bond vector
        
        If tare_dihedrals is True (default False), will also ensure that the dihedral planes of the two Connectors are coplanar
        this may be desirable in many cases, but comes with stricter preconditions, namely both connectors having tangents define
        '''
        bond_antialignment : Rotation = alignment_rotation(self.unit_bond_vector, -other.unit_bond_vector)
        if tare_dihedrals:
            tangent_alignment = alignment_rotation(bond_antialignment.apply(self.tangent_vector), other.tangent_vector)
        else:
            tangent_alignment = Rotation.identity()
        
        return ( # order of application of operations reads bottom-to-top (rightmost operator acts first)
            RigidTransform.from_translation(other.linker.position)
            * RigidTransform.from_rotation(tangent_alignment)
            * RigidTransform.from_rotation(bond_antialignment)
            * RigidTransform.from_translation(-self.anchor.position)
        )

    def antialign_rigidly_to(
        self,
        other : 'Connector',
        tare_dihedrals : bool=False,
        dihedral_angle_rad : Optional[float]=None,
        match_bond_length : bool=False,
    ) -> None:
        '''Align this Connector rigidly to another Connector, based on the calculated rigid alignment transform'''
        self.rigidly_transform(transformation=self.rigid_antialignment_to(other, tare_dihedrals=tare_dihedrals))
        if match_bond_length: 
            self.set_bond_length(other.bond_length) # ensure bond length matches the other Connector
            if (dihedral_angle_rad is not None): # NOTE: sentinel (rather than default 0.0) weakens preconditions on tangents when no dihedral is specified
                self.assign_dihedral(other, dihedral_angle_rad=dihedral_angle_rad)

    def antialigned_rigidly_to(
        self,
        other : 'Connector',
        tare_dihedrals : bool=False,
        dihedral_angle_rad : Optional[float]=None,
        match_bond_length : bool=False,
    ) -> 'Connector':
        '''Return a copy of this Connector rigidly aligned to another Connector'''
        new_connector = self.copy()
        new_connector.antialign_rigidly_to(
            other,
            tare_dihedrals=tare_dihedrals,
            dihedral_angle_rad=dihedral_angle_rad,
            match_bond_length=match_bond_length,
        )

        return new_connector
    
    ## Ballistic alignment
    def ballistic_antialignment_to(self, other : 'Connector') -> RigidTransform:
        '''
        Compute a rigid transformation which aligns a pair of Connectors by turning
        the bond vector of this Connector to face the linker point of other Connector
        The anchor positions of either Connector will be unaffected
        
        Called "ballistic" because the action (especially when matching bond length)
        resembles this Connector aiming and then "shooting" its linker at the other Connector
        '''
        return (
            RigidTransform.from_translation(self.anchor.position)
            * RigidTransform.from_rotation(alignment_rotation(self.bond_vector, other.anchor.position - self.anchor.position))
            * RigidTransform.from_translation(-self.anchor.position)
        )
    
    def antialign_ballistically_to(
        self,
        other : 'Connector',
        match_bond_length : bool=False,
    ) -> None:
        '''
        Match linker position of this Connector to the anchor position of the other Connector (if assigned)
        NOTE: does NOT modify the other Connector, only acts on the first Connector of the provided pair
        '''
        self.rigidly_transform(transformation=self.ballistic_antialignment_to(other))
        if match_bond_length:
            self.set_bond_length(np.linalg.norm(other.anchor.position - self.anchor.position))

    def antialigned_ballistically_to(
        self,
        other : 'Connector',
        match_bond_length : bool=False,
    ) -> None:
        '''
        Return copy of this Connector whose linker positions is aligned to the anchor position of the other Connector (if assigned)
        NOTE: does NOT modify either Connector of the passed pair; returns a modified copy of the first Connector
        '''
        new_connector = self.copy() # DEV: opted not to go for self.rigidly_transformed(self.alignment_transform(...)) to avoid duplicating logic
        new_connector.antialign_ballistically_to(other, match_bond_length=match_bond_length)
        
        return new_connector
    
    ### DEV: asymmetry relative to rigid alignment viz dihedral angles is no accident;
    ### Rigid alignment results in antialignment after one application with bond length matching,
    ### whereas ballistic alignment in general requires both Connecters to be mutually transformed to guarantee antialignment
    def mutually_antialign_ballistically(
        self,
        other : 'Connector',
        dihedral_angle_rad : Optional[float]=None,
    ) -> None:
        '''
        Ballistically align this Connector to the other, and vice-versa
        In the end, the linker of either Connector with be coincident with the
        anchor of the other, and the anchors sites will not have been moved

        If a dihedral angle is provided, will also rotate this Connector along the mutual bond axis to that angle
        '''
        self.antialign_ballistically_to(other, match_bond_length=True)
        other.antialign_ballistically_to(self, match_bond_length=True)
        if (dihedral_angle_rad is not None): # NOTE: sentinel (rather than default 0.0) weakens preconditions on tangents when no dihedral is specified
            self.assign_dihedral(other, dihedral_angle_rad=dihedral_angle_rad)


    # Comparison methods
    def bondable_with(self, other : 'Connector') -> bool:
        '''Whether this Connector is bondable with another Connector instance'''
        if not isinstance(other, Connector):
            return False # DEVNOTE: raise TypeError instead (or at least log a warning)?
        
        # DEV: opting for loosest possible comparison where at least on of the attachable elements overlaps between opposing pairs of attachment points
        # opted not to check the (perhaps more obvious) "self.anchor.attachment in other.linker.attachables", etc., 
        # because the attachment labels may be unassigned between resolution shift operations in the representation hierarchy
        return ( 
            (not set.isdisjoint(self.anchor.attachables, other.linker.attachables))
            and (not set.isdisjoint(self.linker.attachables, other.anchor.attachables))
            and (self.bondtype == other.bondtype)
            # TODO: also compare positions, if set?
        )
        
    def bondable_with_iter(self, *others : Iterable[Union['Connector', Iterable['Connector']]]) -> Generator[bool, None, None]:
        '''Whether this Connector can be connected to each of a sequence of other Connectors, in the order passed'''
        for other in others:
            if isinstance(other, Connector):
                yield self.bondable_with(other)
            elif isinstance(other, Iterable):
                # DEVNOTE: deliberately NOT using "yield from" to preserve parity with input
                # (output element corresponding to iterable is now just a Generator instance, rather than a bool)
                yield self.bondable_with_iter(*other)
            else:
                raise TypeError(f'Connector can only be bonded to other Connectors or collection of Connectors, not with object of type {type(other)}')

    def coincides_with(self, other : 'Connector') -> bool:
        '''Whether this Connector overlaps spatially with another Connector'''
        return ( # TODO: set atol/rtol for float vector comparison
            compare_optional_positions(self.anchor.position, other.anchor.position)
            and compare_optional_positions(self.linker.position, other.linker.position)
            and compare_optional_positions(self._tangent_position, other._tangent_position)
        )

    def resembles(self, other: 'Connector') -> bool:
        '''Whether this Connector has interchangeable component labels (not necessarily positions) with to another Connector'''
        return (
            # and self.anchor.attachment == other.anchor.attachment
            self.anchor.attachables == other.anchor.attachables
            # and self.linker.attachment == other.linker.attachment
            and self.linker.attachables == other.linker.attachables
            and self.bondtype == other.bondtype
        )

    def fungible_with(self, other : 'Connector') -> bool:
        '''Whether this connector can replace other without any change to programs which involve it'''
        return self.coincides_with(other) and self.resembles(other)

    ## Labelling and representation methods
    @property
    def label(self) -> ConnectorLabel:
        '''Identifying label for this Connector'''
        return self._label
    
    @label.setter
    def label(self, new_label : ConnectorLabel) -> None:
        '''Set label for this Connector'''
        if not isinstance(new_label, Hashable):
            raise TypeError(f'Connector label must be a Hashable type, not {type(new_label)}')
        self._label = new_label
    
    def canonical_form(self) -> BondType:
        '''Return a canonical form used to distinguish equivalent Connectors'''
        return self.bondtype # TODO: make this more descriptive; good enough for now

    def __repr__(self) -> str:
        repr_attr_strs : dict[str, str] = {
            'anchor' : self.anchor,
            'linker' : self.linker,
            'bondtype' : self.bondtype,
            'bond_length' : self.bond_length if self.has_bond_vector else None,
            # 'query_smarts' : self.query_smarts,
            'label' : self.label,
            'dihedral_plane_set' : self.has_dihedral_orientation,
        }
        attr_str = ', '.join(
            f'{attr}={value!r}'
                for (attr, value) in repr_attr_strs.items()
        )
        
        return f'{self.__class__.__name__}({attr_str})'

    # def __hash__(self) -> int:
    #     return hash((
    #         # id(self),
    #         self.anchor,
    #         # self.linker,
    #         frozenset(self.linkables), # TODO: make linkables frozen at __init__ level to avoid post-init mutation?
    #         *self.is_position_assigned.keys(),
    #     ))
    #     # raise NotImplementedError # DEVNOTE: need to decide what info should (and shouldn't) go into the making of this sausage
    
    # def __eq__(self, other : 'Connector') -> bool:
    #     # return hash(self) == hash(other)
    #     return self.fungible_with(other)
    
    # Copying and attr transfer methods
    def individualize(self) -> dict[tuple[AttachmentLabel, AttachmentLabel], 'Connector']:
        '''
        Expand a Connector into a set of Connectors with identical properties but 
        distinct, singletons linkables, one for each linkable in the original Connector
        '''
        indiv_conn_map = dict()
        for anchor_label, linker_label in cartesian(self.anchor.attachables, self.linker.attachables):
            conn_clone = self.copy()
            conn_clone.anchor.attachment = anchor_label
            conn_clone.anchor.attachables = {anchor_label}
            
            conn_clone.linker.attachment = linker_label
            conn_clone.linker.attachables = {linker_label}

            indiv_conn_map[(anchor_label, linker_label)] = conn_clone
        return indiv_conn_map
    
    def counterpart(self) -> 'Connector':
        '''
        Create a counterpart Connector which is identical to this Connector but has its linker and anchor sites swapped
        
        By construction, the counterpart will always be bondable with this Connector (and vice versa),
        assuming the attachables set of the anchor and linker point are both non-empty
        '''
        counterpart = self.copy()
        counterpart.anchor, counterpart.linker = self.linker, self.anchor
        if self.has_tangent_position:
            # NOTE: since vector if defined by difference to tangent point, updated tangent 
            # point can be set directly from this difference, since anchor is updated about
            counterpart.tangent_vector = self.tangent_vector 
        
        return counterpart

## Selection between pairs of Connectors (useful, for example, for resolution-shift operations)
ConnectorSelector : TypeAlias = Callable[[Connector, Connector], Connector]

def select_first(connector1 : Connector, connector2 : Connector) -> Connector:
    '''Select the first of a pair of Connectors'''
    return connector1

def select_second(connector1 : Connector, connector2 : Connector) -> Connector:
    '''Select the second of a pair of Connectors'''
    return connector2

def make_second_resemble_first(connector1 : Connector, connector2 : Connector) -> Connector:
    '''Select the first of a pair of Connectors, but merge their linkables'''
    new_connector = connector2.copy()
    new_connector.anchor.attachables.update(connector1.anchor.attachables)
    new_connector.linker.attachables.update(connector1.linker.attachables)
    
    return new_connector

# DEV: provide implementations which make some attempt to reconcile spatial info attache to respective Connectors
...