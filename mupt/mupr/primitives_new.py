'''A refactored version of .primitives which is more cohesive and cleanly adheres to functionality where needed'''

from typing import (
    Any,
    Collection,
    Hashable,
    Iterable,
    Optional,
    Protocol,
    Mapping,
    TypeAlias,
    TypeVar,
)
ConnectorAddress = int
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)
PrimitiveConnectorReference = tuple[PrimitiveHandle, ConnectorAddress]

from abc import ABC, abstractmethod
from copy import deepcopy

from anytree import NodeMixin
from scipy.spatial.transform import RigidTransform

from .canonicalize import lex_order_multiset_str
from .connection import (
    Connector,
    make_second_resemble_first,
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
from .topology import TopologicalStructure, GraphLayout
from .embedding import infer_connections_from_topology, flexible_connector_reference

from ..mutils.containers import UniqueRegistry
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import RigidlyTransformable
from ..chemistry.core import ElementLike, isatom, BOND_ORDER, valence_allowed


# Custom Exceptions
class IrreducibilityError(AttributeError):
    '''Raised when attempting to perform a composite Primitive operation on a simple one'''
    pass

class BijectionError(ValueError):
    '''Raised when a pair of objects expected to be in 1-to-1 correspondence are mismatched'''
    pass

# Protocols
class ManagesConnections(Protocol):
    '''An object which contains Connectors, either internally-paired or externally-facing'''
    connectors : Mapping[ConnectorAddress, Connector]
    connections : Iterable[frozenset[PrimitiveConnectorReference, PrimitiveConnectorReference]]
    internal_connector_addrs : Collection[ConnectorAddress]
    external_connector_addrs : Collection[ConnectorAddress]
    
    def fetch_connector(self, conn_addr : ConnectorAddress) -> Connector:
        ...
    
# Primitive types
class BasePrimitive(  # DEV: eventually, rename to just "Primitive" - temp name for refactoring
    NodeMixin,
    ManagesConnections,
    RigidlyTransformable
):
    '''
    A fundamental, scale-agnostic building block of a molecular system, as represented my MuPT
    '''
    def __init__(
        self,
        shape : Optional[BoundedShape]=None,
        label : PrimitiveLabel='PRIM',
        metadata : Optional[dict[Hashable, Any]]=None, # DEV: make into implicit kwargs instead?
    ) -> None:
        self.shape = shape
        self.label = label
        self.metadata = metadata or dict()
        
        # declaration of attributes - TODO: find better way to typehint these should be defined w/o proiding definition here
        self.topology : TopologicalStructure 

    # TODO: placeholder for connection info?

    # Properties derived from stipulated core pieces of information
    @property
    def functionality(self) -> int:
        return len(self.external_connector_addrs)
    
    @property
    def valence(self) -> int:
        '''Electronic valence of the Primitive, i.e. the total bond order of all external-facing Connectors on this Primitive'''
        total_bond_order : float = sum(
            BOND_ORDER.get(conn.bondtype, 0.0)
                for conn in self.connectors.values()
        )
        return round(total_bond_order)
    chemical_valence = electronic_valence = valence # aliases for convenience
    
    # Geometry
    ## Shape
    @property
    def has_shape(self) -> bool:
        '''Whether this Primitive has an associated external shape'''
        return self._shape is not None
    
    @property
    def shape(self) -> Optional[BoundedShape]: # TODO: make ShapedPrimitive subtype to avoid all these None checks?
        '''The external shape of this Primitive'''
        return self._shape
    
    @shape.setter
    def shape(self, new_shape : Optional[BoundedShape]) -> None:
        '''Set the external shape of this Primitive with another BoundedShape'''
        # Case 1) no shape
        if new_shape is None:
            self._shape = None
            return
        
        # Case 2) valid shape, which may need to have transformation history transferred over
        if not isinstance(new_shape, BoundedShape):
            raise TypeError(f'Primitive shape must be BoundedShape instance, not object of type {type(new_shape.__name__)}')

        new_shape_clone = new_shape.copy() # NOTE: make copy to avoid mutating original (per Principle of Least Astonishment)
        if self._shape is not None:
            new_shape_clone.cumulative_transformation = self._shape.cumulative_transformation # transfer translation history BEFORE overwriting
        
        self._shape = new_shape_clone
        
    ## Applying rigid transformations (fulfilling RigidlyTransformable contracts)
    def _copy_untransformed(self) -> 'BasePrimitive':
        '''Return a new Primitive with the same information and children as this one, but which has no parent'''
        return self.__class__( # DEV: needs augmentation when called on subtypes to get additional info to transfer correctly
            shape=(None if self.shape is None else self.shape.copy()),
            label=deepcopy(self.label),
            metadata=deepcopy(self.metadata),
        )
        
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        '''Apply a rigid transformation to all parts of a Primitive which support it'''
        if isinstance(self.shape, BoundedShape):
            self.shape.rigidly_transform(transformation)
            
        for connector in self.connectors.values():
            connector.rigidly_transform(transformation)
    
## Simples
class SimplePrimitive(BasePrimitive):
    '''
    A Primitive with no internal structure (i.e. no children, topology, or internal connections)
    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    '''
    def __init__(
        self,
        connectors : Optional[Iterable[Connector]]=None, # only entry point into hierarchy (i.e. can't add or remove Connectors to Composites)
        shape : Optional[BoundedShape]=None,
        label : PrimitiveLabel='SIMPLE',
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        super().__init__(
            shape=shape,
            label=label,
            metadata=metadata,
        )
        self._connectors : dict[ConnectorAddress, Connector] = {
            id(conn) : conn
                for conn in connectors
        }
        
    @property
    def connectors(self) -> Mapping[ConnectorAddress, Connector]:
        '''All Connectors accessible from this Primitive'''
        return self._connectors
    
    @property
    def topology(self) -> TopologicalStructure:
        '''
        Always return the trivial (empty) topology for simples
        Acts as a "calling card" for an eventual canonical form for the topology graph
        '''
        return TopologicalStructure()
        
    # Attachment (or lack thereof) of child primitives
    ## TODO: include an explicit reference to attaching a parent above this Simple (OK to do)
    def _pre_attach_children(self, children : Iterable[BasePrimitive]) -> None:
        raise IrreducibilityError('Cannot attach child Primitives to a SimplePrimitive instance')

    def _pre_detach_children(self, children : Iterable[BasePrimitive]) -> None:
        raise IrreducibilityError('Cannot attach child Primitives to a SimplePrimitive instance')
    
class AtomicPrimitive(SimplePrimitive):
    '''
    A Primitive representing a single atom from the periodic table
    Contains element, formal charge, and nuclear mass information about the atom
    '''
    def __init__(
        self,
        element : ElementLike,
        connectors : Optional[Iterable[Connector]]=None,
        shape : Optional[BoundedShape]=None,
        label : PrimitiveLabel='ATOM',
        metadata : Optional[dict]=None,
    ):
        if not isatom(element):
            raise TypeError(f'Invalid element type {type(element)}')
        self._element = element
        
        super().__init__(
            shape=shape,
            connectors=connectors,
            label=label,
            metadata=metadata,
        )

    @property # DEV: no setter implemented; element is immutable after instantiation
    def element(self) -> Optional[ElementLike]:
        '''
        The chemical element, ion, or isotope associated with this AtomicPrimitive
        '''
        return self._element
    
    def check_valence(self) -> None: # DEV: deliberately put this here (i.e. not next to "valence" def) for eventual peelaway when splitting off AtomicPrimitive
        '''Check that element assigned to atomic Primitives and bond orders of Connectors are chemically-compatible'''
        if not self.is_atom:
            return

        if not valence_allowed(self.element.number, self.element.charge, self.valence):
            raise ValueError(f'Atomic {self._repr_brief(include_functionality=True)} with total valence {self.valence} incompatible with assigned element {self.element!r}')
    
    
## Composites
class CompositePrimitive(BasePrimitive):
    '''
    A Primitive with an internal structure of "child" Primitives within it
    Internal attributes about children, their Connectors, and the Topology connecting are immutable after instantiation

    CompositePrimitives form the branches of the a representation hierarchy tree
    '''
    def __init__(
        self,
        children : Iterable[BasePrimitive],
        internal_connections : Iterable[frozenset[PrimitiveConnectorReference, PrimitiveConnectorReference]],
        topology : TopologicalStructure,
        shape : Optional[BoundedShape]=None,
        label : Hashable=None,
        metadata : Optional[dict]=None, 
    ):
        # TODO: check bijection between children and topology on init
        # TODO: include pre-registered handles?
        self.topology = topology # call validator on first-time pass
        self.children_by_handle : UniqueRegistry[PrimitiveHandle, BasePrimitive] = UniqueRegistry()
        self.children_by_handle.register_from(children)

        self._internal_connections = internal_connections
        # TODO: bind all passed connectors as external
        self.external_connectors = dict()
        
        super().__init__(
            shape=shape,
            label=label,
            metadata=metadata,
        )
        
    @property
    def connectors(self) -> Mapping[ConnectorAddress, Connector]:
        '''All Connectors accessible from this Primitive - aggregate of all Connectors on children'''
        return self._connectors
        
class MutableCompositePrimitive(BasePrimitive):
    '''
    A CompositePrimitive which allows for dynamic modification of its internal structure
    (i.e. adding/removing children and connections at will)
    '''
    def __init__( # DEV: could omit entirely; repeated for documentation purposes, and in case extra init config needs to be addeds
        self,
        children : Optional[Iterable[BasePrimitive]]=None,
        topology : Optional[TopologicalStructure]=None,
        shape : Optional[BoundedShape]=None,
        connectors : Optional[Iterable[Connector]]=None,
        label : Hashable=None,
        metadata : Optional[dict]=None, 
    ):
        if children is None:
            children = []

        if topology is None:
            topology = self.compatible_indiscrete_topology()
        
        super().__init__(
            children=children,
            topology=topology,
            shape=shape,
            connectors=connectors,
            label=label,
            metadata=metadata,
        )
        
    def freeze(self) -> CompositePrimitive:
        '''
        Return an immutable CompositePrimitive copy of this MutableCompositePrimitive
        '''
        ...
        
## Builder interface (for acutally setting up hierarchies)
class CompositePrimitiveBuilder(ABC):
    '''
    Class which configures and build Composite hierarchies before freezing them
    Allows construction and usage of repr to be separate while not impinging on immutability of CompositePrimitives
    '''
    @abstractmethod
    def build(self) -> CompositePrimitive:
        ...
        
## Checker methods
def is_atom(prim : BasePrimitive) -> bool:
    '''Check whether a Primitive is an AtomicPrimitive'''
    return isinstance(prim, AtomicPrimitive)

def is_atomizable(prim : BasePrimitive) -> bool:
    '''Check whether a Primitive is either an AtomicPrimitive or a CompositePrimitive which can be fully expanded into AtomicPrimitives'''
    if is_atom(prim):
        return True
    
    if not isinstance(prim, CompositePrimitive):
        return False
    
    return all(
        is_atomizable(child)
            for child in prim.children
    )