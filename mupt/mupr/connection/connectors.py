'''
Core components of connections, namely:
* AttachmentPoints, which define geometric positions and selectivity of attachment sites
* Connectors, which comprise 2 attachment points (an "anchor" and a "linker") and represent half of a chemical bond
'''

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Iterable,
    Optional,
    TypeAlias,
    TYPE_CHECKING,
)

from dataclasses import dataclass, field
from copy import deepcopy
from itertools import product as cartesian
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

if TYPE_CHECKING:
    from .management import HoldsConnectors
from .types import (
    AttachmentLabel,
    ConnectorLabel,
    ConnectorHandle,
)
from .alignment import are_antialigned
from .exceptions import IncompatibleConnectorError
from .types import AttachmentLabel, ConnectorLabel, ConnectorHandle

from ..canonicalize import lex_order_multiset_str
from ...chemistry.core import BondType, BOND_ORDER
from ...geometry.arraytypes import Vector3, Array3x3, as_n_vector
from ...geometry.measure import compare_optional_positions
from ...geometry.coordinates.basis import is_orthonormal
from ...geometry.transforms.linear import rejector
from ...geometry.transforms.rigid.rotations import alignment_rotation
from ...geometry.transforms.rigid.application import RigidlyTransformable


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
            value = as_n_vector(value, dimension=3)
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
        self.anchor : AttachmentPoint = anchor if (anchor is not None) else AttachmentPoint()
        self.linker : AttachmentPoint = linker if (linker is not None) else AttachmentPoint()
        
        self.bondtype : BondType = bondtype
        self.query_smarts : str = query_smarts
        self.label : Hashable = self.__class__.DEFAULT_LABEL if (label is None) else label
        self.metadata : dict[Hashable, Any] = metadata or dict()

        ## Protected attributes
        self._address = uuid4()  # randomly-generated; may opt for field based (with something like uuid7) in the future
        self._address_str = str(self._address) # cache to avoid recalculation

        self._neighbor : Optional[Connector] = None
        self._locked : bool = False
        self._holder : Optional['HoldsConnectors'] = None
        self._tangent_position = None # DEV: no call to setter; must be assigned via protected tangent_vector property

    @property
    def bond_order(self) -> float:
        '''
        A numerical bond order corresponding to the type of bond this Connector is associated with
        E.g. UNASSIGNED = 0.0, AROMATIC = 1.5, DOUBLE = 2.0, etc.
        '''
        return BOND_ORDER.get(self.bondtype, 0.0)

    # Geometric properties
    ## DEV: implemented vector properties (e.g. bond/tangent/normal) by tracking endpoint positions under the hood to get them to
    ## preserving relative orientations for local orthogonal basis under general rigid transformations; key observation is that 
    ## a DIFFERENCE between positions is invariant under shifts of the origin, i.e. if v = (a - b), Tv = T(a - b) = T(a) - T(b), 
    
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
        self.linker.position = as_n_vector(new_bond_vector, dimension=3) + self.anchor.position
        
    @property
    def bond_length(self) -> np.floating:
        '''Distance spanned by the bond vector - i.e. distance from anchor to linker positions'''
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> Vector3:
        '''Unit vector in the same direction as the bond (oriented from anchor to linker)'''
        return self.bond_vector / self.bond_length # DEV: use normalized()?
    
    def set_bond_length(self, new_bond_length : float | np.floating) -> None:
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
        new_tangent_vector = as_n_vector(new_tangent_vector, dimension=3)
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
    
    def local_orthonormal_basis(self) -> Array3x3:
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
        
        if not self.is_antialigned(other, within=alignment_tolerance):
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
            new_connector.tangent_vector = as_n_vector(self.tangent_vector, dimension=3)

        return new_connector

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.anchor.rigidly_transform(transformation)
        self.linker.rigidly_transform(transformation)
        if self.has_tangent_position:
            self._tangent_position = transformation.apply(self._tangent_position)

    # Holder - higher-level object which "holds" this Connector (e.g. for reverse-lookup)
    def has_holder(self) -> bool:
        '''Check if holder has been assigned'''
        return self._holder is not None
    
    @property
    def holder(self) -> Optional['HoldsConnectors']:
        return self._holder
    
    @holder.setter
    def holder(self, new_holder : 'HoldsConnectors') -> None:
        if self._locked:
            raise PermissionError(f'Cannot assign new holder to locked Connector {self}')
        self._holder = new_holder

    @holder.deleter
    def holder(self) -> None:
        if self._locked and not self.has_holder:
            raise PermissionError(f'Cannot remove holder of locked Connector {self}')
        self._holder = None

    # Interactions with neighboring Connectors
    ## Comparison methods
    def bondable_with(self, other : 'Connector') -> bool:
        '''Whether this Connector is bondable with another Connector instance'''
        if not isinstance(other, Connector):
            return False # DEVNOTE: raise TypeError instead (or at least log a warning)?
        # DEV: opting for loosest possible comparison where at least on of the attachable elements overlaps between opposing pairs of attachment points
        # opted not to check the (perhaps more obvious) "self.anchor.attachment in other.linker.attachables", etc., 
        # because the attachment labels may differ between resolution shift operations in the representation hierarchy
        return ( 
            (not set.isdisjoint(self.anchor.attachables, other.linker.attachables))
            and (not set.isdisjoint(self.linker.attachables, other.anchor.attachables))
            and (self.bondtype == other.bondtype)
        )
        
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

    ## Neighbor configuration
    def is_antialigned(self, other : 'Connector', within : float=1E-6) -> bool:
        '''
        Whether this Connector is anti-aligned with another Connector, i.e. whether 
        the anchor of this Connector is within some cutoff distance of the linker
        of the other Connector, and vice-versa (with the same tolerance for both)
        '''
        return are_antialigned(self, other, within=within)
    
    @property
    def is_locked(self) -> bool:
        '''Whether editing of neighbors is allowed'''
        return self._locked

    def lock(self) -> None:
        '''Block editing of neighbors'''
        self._locked = True

    def unlock(self) -> None:
        '''Allow editing of neighbors'''
        self._locked = False
    
    @property
    def has_neighbor(self) -> bool:
        return self._neighbor is not None

    @property
    def neighbor(self) -> Optional['Connector']:
        '''
        The Connector assigned to be this Connector's neighbor, if assigned
        If unassigned, returns None
        '''
        return self._neighbor

    @neighbor.setter
    def neighbor(self, other : 'Connector') -> None:
        if self.is_locked:
            raise PermissionError('Neighbor of this Connector is locked and cannot be modified')

        if not self.bondable_with(other):
            raise IncompatibleConnectorError('Cannot make incompatible Connector neighbor')

        # N.B.: if ALL positions are unset, will evaluate as antialigned
        if not self.is_antialigned(other): # TB: may relax this / allow passing alignment strategy
            raise IncompatibleConnectorError('Candidate for neighbor Connector is not anti-aligne within tolerance')
        
        self._neighbor = other

    @neighbor.deleter
    def neighbor(self) -> None:
        if self.has_neighbor and self.is_locked:
            raise PermissionError('Neighbor of this Connector is locked and cannot be cleared')
        self._neighbor = None

    ## Copying and attr transfer methods
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

    # Labelling and representation methods
    @property
    def address(self) -> str: # protected - no setter or deleter offered
         # opting for str conversion to avoid consumers needing to know about UUID type
        '''
        Hashable address UNIQUE to this Connector
        Not the same as __hash__ (Connector instances with the same hash will have different addresses)
        '''
        return self._address_str
    addr = address # alias for convenience

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

def canonical_form_connectors(
    connectors: Iterable[Connector],
    separator : str=':',
    joiner : str='-',
) -> str:
    '''A hashable string representing a collection of Connectors in canonical form'''
    return lex_order_multiset_str(
        map(Connector.canonical_form, connectors),
        element_repr=Connector.canonical_form,
        separator=separator,
        joiner=joiner,
    )

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