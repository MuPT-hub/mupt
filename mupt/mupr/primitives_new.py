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
class Connectable(Protocol):
    '''An object which contains Connectors, either internally-paired or externally-facing'''
    connectors : Mapping[ConnectorAddress, Connector]
    internal_connections : Iterable[frozenset[PrimitiveConnectorReference, PrimitiveConnectorReference]]
    external_connectors : Collection[ConnectorAddress]
    
# Primitive types
class BasePrimitive(  # DEV: eventually, rename to just "Primitive" - temp name for refactoring
    Connectable,
    NodeMixin,
    RigidlyTransformable
):
    '''
    A fundamental, scale-agnostic building block of a molecular system, as represented my MuPT
    '''
    def __init__(
        self,
        shape : Optional[BoundedShape]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        self._shape = None
        if shape is not None:
            self.shape = shape
               
        self.label = label
        self.metadata = metadata or dict()

    @property
    def functionality(self) -> int:
        return len(self.external_connectors)
    
## Simples
class SimplePrimitive(BasePrimitive):
    '''
    A Primitive with no internal structure (i.e. no children, topology, or internal connections)
    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    '''
    def __init__( # DEV: could omit entirely for now; repeated for documentation purposes, and in case extra init config needs to be addeds
        self,
        connectors : Optional[Iterable[Connector]]=None, # only entry point into hierarchy (i.e. can't add or remove Connectors to Composites)
        shape : Optional[BoundedShape]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        # DEV: include topology init? (will be empty, trivial topology in all cases)
        super().__init__(
            shape=shape,
            label=label,
            metadata=metadata,
        )
        # TODO: register passed connectors
        
    def _pre_attach_children(self, children : Iterable[BasePrimitive]) -> None:
        raise IrreducibilityError('Cannot attach children to SimplePrimitive instances')

    def _pre_detach_children(self, children : Iterable[BasePrimitive]) -> None:
        raise IrreducibilityError('Cannot attach children to SimplePrimitive instances')
    
        
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
        label : Hashable=None,
        metadata : Optional[dict]=None,
    ):
        super().__init__(
            shape=shape,
            connectors=connectors,
            label=label,
            metadata=metadata,
        )

        if not isatom(element):
            raise TypeError(f'Invalid element type {type(element)}')
        self._element = element

    @property # DEV: no setter implemented; element is immutable after instantiation
    def element(self) -> Optional[ElementLike]:
        '''
        The chemical element, ion, or isotope associated with this AtomicPrimitive
        '''
        return self._element
    
## Composites
class CompositePrimitive(BasePrimitive):
    '''
    A Primitive with an internal structure of "child" Primitives within it
    Internal attributes about children, their Connectors, and the Topology connecting are immutable after instantiation

    CompositePrimitives form the branches of the a representation hierarchy tree
    '''
    def __init__(
        self,
        children : Iterable[Primitive],
        internal_connections : Iterable[frozenset[ConnectorReference]],
        topology : TopologicalStructure,
        shape : Optional[BoundedShape]=None,
        label : Hashable=None,
        metadata : Optional[dict]=None, 
    ):
        # TODO: check bijection between children and topology on init
        # TODO: include pre-registered handles?
        self.topology = topology # call validator on first-time pass
        self.children_by_handle : UniqueRegistry[PrimitiveHandle, Primitive] = UniqueRegistry()
        self.children_by_handle.register_from(children)

        self.internal_connections : set[frozenset[ConnectorReference]] = set()
        # TODO: bind all passed connectors as external
        self.external_connectors : dict[ConnectorHandle, ConnectorReference] = dict()
        
    @property
    def connectors(self) -> Mapping[ConnectorAddress, Connector]:
        '''All Connectors accessible from this Primitive - aggregate of all Connectors on children'''
        
class MutableCompositePrimitive(BasePrimitive):
    '''
    A CompositePrimitive which allows for dynamic modification of its internal structure
    (i.e. adding/removing children and connections at will)
    '''
    def __init__( # DEV: could omit entirely; repeated for documentation purposes, and in case extra init config needs to be addeds
        self,
        children : Optional[Iterable[Primitive]]=None,
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