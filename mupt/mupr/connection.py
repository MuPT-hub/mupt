'''Abstractions of connections between two primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Any,
    ClassVar,
    Generator,
    Hashable,
    Iterable,
    Literal,
    Optional,
    TypeVar,
)
Shape = tuple # alias for typehinting array shapes
ConnectorLabel = TypeVar('ConnectorLabel', bound=Hashable)

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform
from rdkit.Chem.rdchem import BondType

from ..geometry.measure import normalized
from ..geometry.transforms.linear import rejector
from ..geometry.transforms.rigid.rotations import alignment_rotation
from ..geometry.transforms.rigid.application import RigidlyTransformable


class ConnectionError(Exception):
    '''Raised when Connector-related errors as encountered'''
    pass

class IncompatibleConnectorError(ConnectionError):
    '''Raised when attempting to connect two Connectors which are, for whatever reason, incompatible'''
    pass


@dataclass(frozen=False) # DEVNOTE need to preserve mutability for now, since coordinates of parts may change
class Connector(RigidlyTransformable):
    '''Abstraction of the notion of a chemical bond between a known body (anchor) and an indeterminate neighbor body (linker)'''
    # DEVNOTE: want to hone in on the allowable types for these (Hashable?)
    anchor : Hashable
    linker : Optional[Hashable] = None
    linkables : set[Hashable] = field(default_factory=set)
    
    bondtype : BondType = BondType.UNSPECIFIED
    query_smarts : str = ''
    
    anchor_position  : Optional[np.ndarray[Shape[Literal[3]], float]] = field(default=None, init=True)
    linker_position  : Optional[np.ndarray[Shape[Literal[3]], float]] = field(default=None, init=True)
    tangent_position : Optional[np.ndarray[Shape[Literal[3]], float]] = field(default=None, init=True)

    # Geometric properties
    _POSITION_ATTRS : ClassVar[tuple[str]] = (
        # DEV: this will need updating if more position-type attributes are added; manually curating this is fine for now
        'anchor_position',
        'linker_position',
        'tangent_position',
    ) 
    
    @staticmethod
    def compare_optional_positions(
        position_1 : Optional[np.ndarray[Shape[Any], float]],
        position_2 : Optional[np.ndarray[Shape[Any], float]],
        **kwargs,
    ) -> bool:
        '''Check that two positional attributes are either 1) both undefined, or 2) both defined and equal'''
        if type(position_1) != type(position_2):
            return False
        
        if position_1 is None: # both are None
            return True
        elif isinstance(position_1, np.ndarray):
            return np.allclose(position_1, position_2, **kwargs)
        else:
            raise TypeError(f'Expected position attributes to be either None or numpy.ndarray, got {type(position_1)} and {type(position_2)}')
        
    ## Checks for defined positions - DEV: replace with monadic interface down the line ("Maybe" pattern?)
    @property
    def has_anchor_position(self) -> bool:
        '''Determine whether this Connector has an anchor position (i.e. local position) defined'''
        return self.anchor_position is not None
    
    @property
    def has_linker_position(self) -> bool:
        '''Determine whether this Connector has a linker position (i.e. off-body connection location) defined'''
        return self.linker_position is not None

    @property
    def has_tangent_position(self) -> bool:
        '''Determine whether this Connector has a tangent vector (i.e. dihedral plane orientation) defined'''
        return self.tangent_position is not None
    
    ## bond vector
    @property
    def has_bond_positions(self) -> bool:
        return self.has_anchor_position and self.has_linker_position
    
    @property
    def bond_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        if not self.has_bond_positions:
            raise ValueError
        return self.linker_position - self.anchor_position
    
    @property
    def bond_length(self) -> float:
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''Unit vector in the same direction as the bond (oriented from anchor to linker)'''
        return self.bond_vector / self.bond_length
    
    def set_bond_length(self, new_bond_length : float) -> None:
        '''Move the linker site along the bond axis to a set distance away from the anchor'''
        self.linker_position = new_bond_length*self.unit_bond_vector + self.anchor_position
        
    ## Tangent vector (defines dihedral plane)
    has_defined_dihedral_plane = has_tangent_position # alias tangent position property
    
    @property
    def tangent_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''
        Vector tangent to the dihedral plane and orthogonal to the bond vector
        
        The tangent and bond vectors span the dihedral plane and 
        fix a local right-handed coordinate system for the Connector
        '''
        if not self.has_defined_dihedral_plane:
            raise ValueError('Connector does not have a dihedral orientation set')
        
        tangent = normalized(self.tangent_position - self.anchor_position) # DEVNOTE: worth providing option to not normalize?
        if not np.isclose(np.dot(self.bond_vector, tangent), 0.0):
            raise ValueError('Badly set tangent position: resultant dihedral plane does not contain this Connector\'s bond vector')
        
        return tangent
    
    @tangent_vector.setter
    def tangent_vector(self, vector : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Update tangent positions given a new tangent vector'''
        # NOTE: implemented this way to get tangent to transform correctly under rigid transformations;
        # a DIFFERENCE between vectors is invariant to shifts of the origin; the same is done for the bond vector
        self.tangent_position = vector + self.anchor_position # DEV: move validation of tangent position orthogonality into here?

    def set_tangent_from_coplanar_point(self, coplanar_point : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Set the dihedral tangent point from a third point in the dihedral plane'''
        self.tangent_vector = rejector(self.bond_vector) @ (coplanar_point - self.anchor_position)

    def set_tangent_from_normal_point(self, normal_point : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Set the dihedral tangent point from a point on the span of the normal to the dihedral plane'''
        self.tangent_vector = np.cross(self.bond_vector, normal_point - self.anchor_position)
        
    @property
    def normal_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''A unit vector normal to the dihedral plane (if assigned)'''
        return normalized( np.cross(self.bond_vector, self.tangent_vector) )
        
    ## Applying rigid transformations
    ### Fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'Connector':
        # return Connector(**self.__dict__)
        new_connector = self.__class__(
            anchor=self.anchor, # TODO: does this need to be deepcopied?
            linker=self.linker,
            linkables=set(linkable for linkable in self.linkables),
            bondtype=self.bondtype,
            query_smarts=self.query_smarts,
        )

        # make copies of positional attribute arrays, if they aren't unset
        for pos_attr in self._POSITION_ATTRS:
            if (position := getattr(self, pos_attr)) is None:
                setattr(new_connector, pos_attr, None) # DEV: opted to make this explicit for clarity, but could be made implicit if desired
            else:
                setattr(new_connector, pos_attr, np.array(position))

        return new_connector

    def _rigidly_transform(self, transform : RigidTransform) -> None:
        for pos_attr in self._POSITION_ATTRS:
            if (position := getattr(self, pos_attr)) is not None:
                setattr(self, pos_attr, transform.apply(position))
    
    ### Aligning Connectors to one another 
    def rigid_alignment_transform_to(
            self,
            other : 'Connector',
            dihedral_angle_rad : float=0.0,
        ) -> RigidTransform:
        '''
        Compute a rigid transformation which aligns a pair of Connectors by making
        the linker point of this Connector coincident with the anchor of the other Connector,
        the Connectors' bond vectors antiparallel, and the Connectors' tangent vectors subtend the
        desired dihedral angle in radians (by default, 0.0 rad)
        
        If the two Connectors have the same bond length, the anchor of this Connector will be coincident with the linker
        of the other; otherwise, the anchor will merely lay on the span of the other Connectors bond vector
        '''
        if not (self.has_bond_positions and other.has_bond_positions):
            raise ValueError('Cannot compute rigid alignment transform with undefined Connector positions')
        
        if not (self.has_defined_dihedral_plane and other.has_defined_dihedral_plane):
            raise ValueError('Cannot compute faithful orientation for rigid alignment transform with undefined Connector orientations')

        ## NOTE: the orthogonality of the tangent and bond vector of each Connector allows the tangent alignment to
        ## not disturb the preceding bond antialignment, and to fix a unique orthogonal change of basis
        bond_antialignment : Rotation = alignment_rotation(self.unit_bond_vector, -other.unit_bond_vector)
        tangent_alignment  : Rotation = alignment_rotation(bond_antialignment.apply(self.tangent_vector), other.tangent_vector)
        dihedral_rotation = Rotation.from_rotvec(dihedral_angle_rad * other.unit_bond_vector)

        return ( # order of application of operations reads bottom-to-top (rightmost operator acts first)
            RigidTransform.from_translation(other.linker_position)
            * RigidTransform.from_rotation(dihedral_rotation)
            * RigidTransform.from_rotation(tangent_alignment)
            * RigidTransform.from_rotation(bond_antialignment)
            * RigidTransform.from_translation(-self.anchor_position)
        )

    def align_rigidly_to(
            self,
            other : 'Connector',
            dihedral_angle_rad : float=0.0,
            match_bond_length : bool=False,
        ) -> None:
        '''Align this Connector rigidly to another Connector, based on the calculated rigid alignment transform'''
        self.rigidly_transform(transformation=self.rigid_alignment_transform_to(other, dihedral_angle_rad))
        if match_bond_length: 
            self.set_bond_length(other.bond_length) # ensure bond length matches the other Connector

    def aligned_rigidly_to(
            self,
            other : 'Connector',
            dihedral_angle_rad : float=0.0,
            match_bond_length : bool=False,
        ) -> 'Connector':
        '''Return a copy of this Connector rigidly aligned to another Connector'''
        new_connector = self.copy() # DEV: opted not to go for self.rigidly_transformed(self.alignment_transform(...)) to avoid duplicating logic
        new_connector.align_rigidly_to(
            other,
            dihedral_angle_rad=dihedral_angle_rad,
            match_bond_length=match_bond_length,
        )

        return new_connector
    
    def align_ballistically_to(
        self,
        other : 'Connector',
        match_bond_length : bool=False,
    ) -> None:
        '''
        Match linker position of this Connector to the anchor position of the other Connector (if assigned)
        NOTE: does NOT modify the other Connector, only acts on the first Connector of the provided pair
        '''
        if not other.has_anchor_position:
            raise AttributeError('No target anchor position defined for ballistic alignment')
        self.linker_position = other.anchor_position
        if self.has_tangent_position:
            ...
            # TODO: define how to transfer tangent along new bond vector

    def aligned_ballistically_to(self, other : 'Connector', match_bond_length : bool=False) -> None:
        '''
        Return copy of this Connector whose linker positions is aligned to the anchor position of the other Connector (if assigned)
        NOTE: does NOT modify either Connector of the passed pair; returns a modified copy of the first Connector
        '''
        new_connector = self.copy() # DEV: opted not to go for self.rigidly_transformed(self.alignment_transform(...)) to avoid duplicating logic
        new_connector.align_ballistically_to(other, match_bond_length=match_bond_length)
        
        return new_connector
    
    # DEV: was unsure of whether or not to make this a classmethod; opted for instance method instead, with the understanding
    # that you can still call it like a classmethod (i.e. conn1.align(conn2) <-> Connector.align(conn1, conn2))
    def mutually_align_ballistically(self, other : 'Connector') -> Optional[float]:
        '''
        Ballistically align this Connector to the other, and vice-versa
        In the end, the linker of either Connector with be coincident with the anchor or the other, and the anchors sites will not have been moved
        If dihedral (i.e. tangent)
        
        Returns the dihedral angle resulting between the new aligned Connectors, 
        or None if an explicit dihedral plane orientation is not defined for either
        '''
        # Perform mutual alignment in-place
        self.align_ballistically_to(other)
        other.align_ballistically_to(self)
        
        # Calculate resultant dihedral angle - TODO: implement calculation of this
        if self.has_tangent_position and other.has_tangent_position:
            dihedral_angle_rad = np.arccos(
                np.dot(
                    normalized(self.tangent_vector),
                    normalized(other.tangent_vector),
                )
            )
        else:
            dihedral_angle_rad = None 
        
        return dihedral_angle_rad

    # Comparison methods
    def bondable_with(self, other : 'Connector') -> bool:
        '''Whether this Connector is bondable with another Connector instance'''
        if not isinstance(other, Connector):
            return False # DEVNOTE: raise TypeError instead (or at least log a warning)?
        
        return (
            (self.anchor in other.linkables)
            and (other.anchor in self.linkables)
            and (self.bondtype == other.bondtype)
            # TODO: also compare positions, if set?
        )
        
    def bondable_with_iter(self, *others : Iterable['Connector']) -> Generator[bool, None, None]:
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
        return all(
            self.compare_optional_positions(
                getattr(self, position_attr),
                getattr(other, position_attr),
            )
                for position_attr in self._POSITION_ATTRS
        )

    def resembles(self, other: 'Connector') -> bool:
        '''Whether this Connector has interchangeable components relative to another Connector'''
        return (
            self.anchor == other.anchor
            # and self.linker == other.linker
            and self.linkables == other.linkables
            and self.bondtype == other.bondtype
        )

    def fungible_with(self, other : 'Connector') -> bool:
        '''Whether this connector can replace other without any change to programs which involve it'''
        return self.coincides_with(other) and self.resembles(other)

    # labelling and representation methods
    @property
    def label(self) -> Hashable:
        '''Unique identifying label for this Connector'''
        return id(self)
    
    def canonical_form(self) -> BondType:
        '''Return a canonical form used to distinguish equivalent Connectors'''
        return self.bondtype # TODO: make this more descriptive; good enough for now

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
