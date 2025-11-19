'''A refactored version of .primitives which is more cohesive and cleanly adheres to functionality where needed'''

from typing import (
    Any,
    AbstractSet, # covers both set and frozenset
    ClassVar,
    Collection,
    Hashable,
    Iterable,
    Optional,
    Protocol,
    Mapping,
    TypeVar,
)
ConnectorAddress = TypeVar('ConnectorAddress', bound=int)
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)
PrimitiveConnectorReference = tuple[PrimitiveHandle, ConnectorAddress]
Connection = AbstractSet[PrimitiveConnectorReference, PrimitiveConnectorReference] # using set, rather than tuple, to avoid order-dependence

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
    connections : Iterable[Connection]
    internal_connector_addrs : Collection[ConnectorAddress]
    external_connector_addrs : Collection[ConnectorAddress]
    
    
# Primitive types
class BasePrimitive(  # DEV: eventually, rename to just "Primitive" - temp name for refactoring
    ABC,
    NodeMixin,
    RigidlyTransformable
):
    '''
    A fundamental, scale-agnostic building block of a molecular system, as represented my MuPT
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'PRIM'
    
    def __init__(
        self,
        shape : Optional[BoundedShape]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : Optional[dict[Hashable, Any]]=None, # DEV: make into implicit kwargs instead?
        # TODO: augment init args?
    ) -> None:
        self.shape = shape
        self.label = label if (label is not None) else self.DEFAULT_LABEL 
        self.metadata = metadata or dict()
        
        # declaration of attributes - TODO: find better way to typehint these should be defined w/o proiding definition here
        self._connectors : Mapping[ConnectorAddress, Connector]
        self._internal_connector_addresses : Collection[ConnectorAddress]
        self._external_connector_addresses : Collection[ConnectorAddress]
        self._topology : TopologicalStructure 

    # Connections
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
    
    ## "Semi-abstract" properties - subtypes implement how "hidden" versions of these attrs are provided - DEV: this seems a little sketchy, revisit later
    @property
    def connectors(self) -> Mapping[ConnectorAddress, Connector]:
        '''All Connectors accessible from this Primitive and any below it in a resolution hierarchy'''
        return self._connectors
    
    @property
    def internal_connector_addresses(self) -> Collection[ConnectorAddress]:
        return self._internal_connector_addresses
    
    @property
    def external_connector_addresses(self) -> Collection[ConnectorAddress]:
        return self._external_connector_addresses
    
    @property
    def topology(self) -> TopologicalStructure:
        return self._topology
    
    ## Local contracts for subtypes
    @abstractmethod
    def fetch_connector(self, conn_addr : ConnectorAddress) -> Connector:
        ...
        
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
            
    # Depiction
    ## Hashable canonical forms for core components
    def canonical_form_connectors(self, separator : str=':', joiner : str='-') -> str:
        '''A canonical string representing this Primitive's Connectors'''
        return lex_order_multiset_str(
            (
                self.connectors[connector_handle].canonical_form()
                    for connector_handle in sorted(self.connectors.keys()) # sort by handle to ensure canonical ordering
            ),
            element_repr=str, #lambda bt : BondType.values[int(bt)]
            separator=separator,
            joiner=joiner,
        )
    
    def canonical_form_shape(self) -> str: # DEVNOTE: for now, this doesn't need to be abstract (just use type of Shapefor all kinds of Primitive)
        '''A canonical string representing this Primitive's shape'''
        return type(self.shape).__name__ # TODO: move this into .shape - should be responsibility of individual Shape subclasses
    
    def canonical_form(self) -> str: # NOTE: deliberately NOT a property to indicated computing this might be expensive
        '''A canonical representation of a Primitive's core parts; induces a natural equivalence relation on Primitives
        I.e. two Primitives having the same canonical form are to be considered interchangable within a polymer system
        '''
        elem_form : str = self.element.symbol if (self.element is not None) else str(None) # TODO: move this to external function, eventually
        return f'{elem_form}' \
            f'({self.canonical_form_connectors()})' \
            f'[shape={self.canonical_form_shape()}]' \
            f'<graph_hash={self.topology.canonical_form()}>'

    def canonical_form_peppered(self) -> str:
        '''
        Return a canonical string representation of the Primitive with peppered metadata
        Used to distinguish two otherwise-equivalent Primitives, e.g. as needed for graph embedding
        
        Named for the cryptography technique of augmenting a hash by some external, stored data
        (as described in https://en.wikipedia.org/wiki/Pepper_(cryptography))
        '''
        return f'{self.canonical_form()}-{self.label}' #{self.metadata}'

    ## Stdout printing
    def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        return self.canonical_form_peppered()
    
    def __repr__(self) -> str:
        raise NotImplementedError # TODO - will likely have to change for subtypes
    
    
## Simples
class SimplePrimitive(BasePrimitive):
    '''
    A Primitive with no internal structure (i.e. no children, topology, or internal connections)
    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'SIMPLE'
    
    def __init__(
        self,
        connectors : Optional[Iterable[Connector]]=None, # only entry point into hierarchy (i.e. can't add or remove Connectors to Composites)
        shape : Optional[BoundedShape]=None,
        label : Optional[PrimitiveLabel]=None,
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
        ...
        self._topology = TopologicalStructure() # trivial topology for simples - "calling card" for an eventual canonical graph form
    
    def fetch_connector(self, conn_addr : ConnectorAddress) -> Connector:
        return self._connectors[conn_addr] # NOTE: deliberately avoiding call via dict.get() to raise loud KeyError
        
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
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'ATOM'
    
    def __init__(
        self,
        element : ElementLike,
        connectors : Optional[Iterable[Connector]]=None,
        shape : Optional[BoundedShape]=None,
        label : Optional[PrimitiveLabel]=None,
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
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'TREE'
    
    # Validators
    @staticmethod
    def check_connections_compatible_with_primitive_registry(
        primitive_registry : UniqueRegistry[PrimitiveHandle, BasePrimitive],
        connections : Iterable[Connection], # DEV: weakened type requirement here, even though in practice this will most like be a set or frozenset
    ) -> None:
        '''
        Check that a collection of connections (i.e. pairs of (PrimitiveHandle, ConnectorAddress) references)
        is absolutely compatible with a handled registry of Primitives
        '''
        for (prim_handle_1, conn_addr_1), (prim_handle_2, conn_addr_2) in connections:
            if prim_handle_1 == prim_handle_2:
                raise ValueError(f'Attempted to connect Primitive with handle "{prim_handle_1}" to itself')
            
            if conn_addr_1 == conn_addr_2:
                raise IncompatibleConnectorError(f'Connections must be between distinct pair of Connector instances, not single Connector at address {conn_addr_1}')
            
            for prim_handle in (prim_handle_1, prim_handle_2):
                if prim_handle not in primitive_registry:
                    raise ValueError(f'Primitive with handle "{prim_handle}" referenced in internal connections but does not exist in provided registry of children')
                
            if not Connector.bondable_with( # NOTE: fetch also implicitly checks each Connector exists on respective child
                primitive_registry[prim_handle_1].fetch_connector(conn_addr_1),
                primitive_registry[prim_handle_2].fetch_connector(conn_addr_2),
            ):
                raise IncompatibleConnectorError(
                    f'Connector {conn_addr_1} on Primitive {prim_handle_1} is not bondable with Connector {conn_addr_2} on Primitive {prim_handle_2}'
                )

    @staticmethod
    def check_primitive_registry_bijective_to_topology_nodes(
        primitive_registry : UniqueRegistry[PrimitiveHandle, BasePrimitive],
        topology : TopologicalStructure,
    ) -> None:
        '''
        Verify 1:1 correspondence between the reference handles in a 
        registry of Primitives and the nodes in an incidence topology
        '''
        num_children : int = len(primitive_registry) # perform cheap counting check first to fail faster
        if topology.number_of_nodes() != num_children:
            raise BijectionError(f'Cannot bijectively map {num_children} child Primitives onto {topology.number_of_nodes()}-element topology')
        
        node_labels = set(topology.nodes)
        child_handles = set(primitive_registry.keys())
        if node_labels != child_handles:
            raise BijectionError(
                f'Set underlying topology does not correspond to handles on child Primitives; {len(node_labels - child_handles)} element(s)'\
                f' present without associated children, and {len(child_handles - node_labels)} child Primitive(s) are unrepresented in the topology'
            )

    @staticmethod
    def check_connections_bijective_to_topology_edges(
        connections : AbstractSet[Connection],
        topology : TopologicalStructure,
    ) -> None:
        '''
        Verify that a 1:1 correspondence exists between the internal connections
        (Connectors paired between sibling child Primitives) and the edges present in the incidence topology
        '''
        num_connections : int = len(connections) # perform cheap counting check first to fail faster
        if (num_edges := topology.number_of_edges()) != num_connections:
            raise BijectionError(f'Cannot bijectively map {num_connections} internal connections onto {num_edges}-edge topology')

        edge_labels = set(frozenset(edge) for edge in topology.edges) # cast to frozenset to remove order-dependence
        if edge_labels != connections:
            raise BijectionError(
                f'Incident pairs in associated topology do not correspond to internally-connected pairs of child Primitives;'\
                f'{len(edge_labels - connections)} edge(s) have no corresponding connection, '\
                f'and {len(connections - edge_labels)} internal connection(s) are unrepresented in the topology'
            )
    
    # Connection management
    @property
    def connections(self) -> AbstractSet[Connection]:
        '''
        Generalization chemical bonds - consists of all internally-connected pairs of Connectors,
        represented by unordered pairs of Primitive handle and Connector references
        '''
        return self._connections
    
    def fetch_connector(self, conn_addr : ConnectorAddress) -> Connector:
        ... # TODO: impl recursively
        
        
class FrozenCompositePrimitive(CompositePrimitive):
    '''
    Composite which is Immutable after instantiation
    Validation checks are front-loaded and property lookups are cached at initialization time
    '''
    def __init__(
        self,
        children : UniqueRegistry[PrimitiveHandle, BasePrimitive],
        connections : Iterable[Connection],
        topology : TopologicalStructure,
        shape : Optional[BoundedShape]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : Optional[dict]=None, 
    ):
        # Validate and extract connection info
        CompositePrimitive.check_connections_compatible_with_primitive_registry(children, connections)
        connectors : dict[ConnectorAddress, Connector] = {
            conn_addr : connector
                for child in children
                    for conn_addr, connector in child.connectors.items()
        }
        
        all_connector_addresses = set(connectors.keys())
        self._internal_connector_addresses : set[ConnectorAddress] = set(
            conn_addr
                for connection in connections
                    for prim_handle, conn_addr in connection
        )
        self._external_connector_addresses : set[ConnectorAddress] = all_connector_addresses - self._internal_connector_addresses # guaranteed valid by above precondition
        
        # Validate and set topology
        CompositePrimitive.check_primitive_registry_bijective_to_topology_nodes(children, topology)
        CompositePrimitive.check_connections_bijective_to_topology_edges(connections, topology)
        self._topology = topology # call validator on first-time pass
        
        # Initialization proper
        self.children_by_handle = children
        for child in children.values():
            child.parent = self # TODO: apply readonly trick (https://anytree.readthedocs.io/en/latest/tricks/readonly.html) to add children first, then make immutable forevermore
            
        
        super().__init__( # DEV: super() call at end to ensure all validations pass first
            shape=shape,
            label=label,
            metadata=metadata,
        )
        
    # cached properties
    ...
        
    # Overriding RigidlyTransformable contracts to apply recursively to children as well
    def _copy_untransformed(self) -> 'BasePrimitive':
        raise NotImplementedError
        
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        raise NotImplementedError
        
class MutableCompositePrimitive(CompositePrimitive): # DEV: this will behave by far the closest to the current Primitive impl
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
        label : Optional[PrimitiveLabel]=None,
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
        
    # Resolution shift operations
    ...
    
    # TODO: implement topology.setter (with reference back to base for getter
        
    def freeze(self) -> FrozenCompositePrimitive:
        '''
        Return an immutable CompositePrimitive copy of this MutableCompositePrimitive
        '''
        ...
        
    # Overriding RigidlyTransformable contracts to apply recursively to children as well
    def _copy_untransformed(self) -> 'BasePrimitive':
        raise NotImplementedError
        
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        raise NotImplementedError
        
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