'''A refactored version of mupr.primitives which is more cohesive and cleanly adheres to functionality where needed'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Any,
    AbstractSet, # covers both set and frozenset
    Callable,
    ClassVar,
    Collection,
    Hashable,
    Iterable,
    Optional,
    Mapping,
    Self,
    TypeVar,
)
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
PrimitiveAddress = Hashable
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)

from copy import deepcopy

from anytree import NodeMixin, findall
import networkx as nx
from scipy.spatial.transform import RigidTransform

from .connection.connectors import Connector, canonical_form_connectors
from .connection.exceptions import (
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
from .connection.types import (
    ManagesConnectors,
    ConnectorAddress,
    Connection,
)
from .topology import GraphLayout, canonical_graph_property
from .embedding import (
    infer_connections_from_topology,
    flexible_connector_reference,
    check_primitive_registry_bijective_to_topology_nodes,
    check_connections_bijective_to_topology_edges,
    check_connections_compatible_with_primitive_registry,
)

from ..mutils.containers import UniqueRegistry
from ..geometry.shapes import Shaped, BoundedTransformableShape
from ..geometry.transforms.rigid import RigidlyTransformable
from ..chemistry.core import ElementLike, isatom, BOND_ORDER, valence_allowed
from ..roles import PrimitiveRole


class IrreducibilityError(AttributeError):
    '''Raised when attempting to perform a composite Primitive operation on a simple one'''
    pass

class AtomicityError(IrreducibilityError):
    '''Raised when attempting to perform a composite Primitive operation on a simple one (or vice-versa)'''
    pass

class MissingSubprimitiveError(KeyError):
    '''Raised when a child Primitive expected for a call is not present'''
    pass

    
# Primitive types        
class Primitive(
    Shaped,
    RigidlyTransformable,
    ManagesConnectors,
):
    '''
    A fundamental, scale-agnostic building block of a molecular system, as represented my MuPT
    '''
    # Attributes
    ## Classwide defaults
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'PRIM'

    # Expected instance attributes
    metadata : dict[Hashable, Any]

    # Derived properties
    @property
    def label(self) -> PrimitiveLabel:
        '''A distinguishing label which can be assigned by the user for identification purposes'''
        if 'label' in self.metadata:
            return self.metadata['label']
        return self.DEFAULT_LABEL
    
    def address(self) -> PrimitiveAddress:
        '''Unique identifier used to identify this Connector instances, irrespective of similarity to other Connectors'''
        ...

    # Geometry
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        '''Apply a rigid transformation to all parts of a Primitive which support it'''
        if isinstance(self.shape, RigidlyTransformable):
            self.shape.rigidly_transform(transformation)
            
        for connector in self.connectors:
            connector.rigidly_transform(transformation)
            
    def _copy_untransformed(self) -> Self:
        return NotImplemented

    # Topology
    def neighbors(self) -> Iterable['Primitive']:
        ...

    def __hash__(self) -> int: # Needs to be implemented for Primitives to be used as nodes in networkx graphs
        ...
        
    # Depiction
    ## Stdout printing
    # def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
    #     return self.canonical_form() # self.canonical_form_salted()
    
    # def __repr__(self) -> str:
    #     raise NotImplementedError # TODO - will likely have to change for subtypes
    
    
## Simples
class SimplePrimitive(Primitive, NodeMixin):
    '''
    A Primitive with no internal structure (i.e. no children, topology, or internal connections)
    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'SIMPLE'
    
    def __init__(
        self,
        connectors : Optional[Iterable[Connector]]=None, # only entry point into hierarchy (i.e. can't add or remove Connectors to Composites)
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        self.connectors = tuple(connectors) if connectors is not None else tuple()
        self.connectors_by_address : dict[ConnectorAddress, Connector] = {
            hash(conn) : conn
                for conn in self.connectors # NOTE: not iterating over connectors directly in case it was an Iterator which was exhausted during self.connectors assignment
        }
        self._shape = shape
        self.metadata = metadata or dict()
    
    # Exposing Connectors
    def connector(self, conn_addr : ConnectorAddress) -> Connector:
        return self.connectors_by_address[conn_addr] # NOTE: deliberately avoiding call via dict.get() to raise loud KeyError when missing
    
    def register_connector(
        self,
        connector : Connector,
        conn_addr : Optional[ConnectorAddress]=None,
    ) -> ConnectorAddress:
        ...

    def deregister_connector(
        self,
        conn_addr : ConnectorAddress,
    ) -> Connector:
        ...
        
    # Attachment (or lack thereof) of child primitives
    def _pre_attach_children(self, children : Iterable[Primitive]) -> None:
        raise IrreducibilityError('Cannot attach child Primitives to a SimplePrimitive instance')

    def _pre_detach_children(self, children : Iterable[Primitive]) -> None:
        raise IrreducibilityError('Cannot attach child Primitives to a SimplePrimitive instance')
    
    # TODO: introduce AttributeError on __getitem__ when requesting .children
    
    def _copy_untransformed(self) -> 'SimplePrimitive':
        '''Return a new Primitive with the same information and children as this one, but which has no parent'''
        return self.__class__( # DEV: needs augmentation when called on subtypes to get additional info to transfer correctly
            connectors=(connector.copy() for connector in self.connectors), # copy to avoid cross-reference
            shape=(None if self.shape is None else self.shape.copy()),
            metadata=deepcopy(self.metadata),
        )
    
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
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict]=None,
    ) -> None:
        if not isatom(element):
            raise TypeError(f'Invalid element type {type(element)}')
        self._element = element
        
        super().__init__(
            connectors=connectors,
            shape=shape,
            metadata=metadata,
        )

    @property # DEV: no setter implemented; element is immutable after instantiation
    def element(self) -> ElementLike:
        '''The chemical element, ion, or isotope associated with this AtomicPrimitive'''
        return self._element
    
    def check_valence(self) -> None:
        '''Check that element assigned to atomic Primitives and bond orders of Connectors are chemically-compatible'''
        if not valence_allowed(self.element.number, self.element.charge, self.valence):
            # raise ValueError(f'Atomic {self._repr_brief(include_functionality=True)} with total valence {self.valence} incompatible with assigned element {self.element!r}')
            raise ValueError(f'Atomic {self!r} with total valence {self.valence} incompatible with assigned element {self.element!r}')
    
    def canonical_form(self) -> str:
        return f'{self.element.symbol}{canonical_form_primitive(self)}'
    
## Composites
class CompositePrimitive(Primitive, NodeMixin):
    '''
    A Primitive with an internal structure of "child" Primitives within it
    Internal attributes about children, their Connectors, and the Topology connecting are immutable after instantiation

    CompositePrimitives form the branches of the a representation hierarchy tree
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'TREE'
    
    children_by_address : Mapping[PrimitiveAddress, Primitive]    
    
    # Inspecting children of Composite
    def child(self, prim_addr : PrimitiveAddress) -> Primitive:
        ... # TODO: provide overload which uses a handle <-> address isomorphism
            
    # Search
    def search_hierarchy_by(
        self,
        condition : Callable[['Primitive'], bool],
        halt_when : Optional[Callable[['Primitive'], bool]]=None,
        to_depth : Optional[int]=None,
        min_count : Optional[int]=None,
        max_count : Optional[int]=None,
    ) -> tuple['Primitive']:
        '''
        Return all Primitives below this one in the hierarchy (not just children,
        but anything below them as well!) which match the provided condition.
        
        Matching descendant Primitives are returned in traversal preorder from the root
        '''
        return findall(
            self,
            filter_=condition,
            stop=halt_when,
            maxlevel=to_depth,
            mincount=min_count,
            maxcount=max_count,
        )
        
    # Topology
    def export_cross_section(self, criterion : Callable[[Primitive], bool]) -> nx.Graph:
        '''Generate a graph of a "slice" of a subset of sub-Primitives specified by a criterion'''
        raise NotImplementedError

class FrozenCompositePrimitive(CompositePrimitive):
    '''
    Composite which is Immutable after instantiation
    Validation checks are front-loaded and property lookups are cached at initialization time
    '''
    def __init__(
        self,
        children : UniqueRegistry[PrimitiveHandle, Primitive],
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict]=None, 
    ) -> None:
        # Validate and extract connection info
        # check_connections_compatible_with_primitive_registry(children, connections)\
        connectors_all : list[Connector] = [] # TODO: make Registries and set labels procedurally (somehow)
        connectors_free : list[Connector] = []
        connectors_bound : list[Connector] = []
        for child in children:
            child.parent = self
            for conn_free in child.connectors_free:
                connectors_free.append(conn_bound)
                connectors_all.append(conn_free)

            for conn_bound in child.connectors_free:
                connectors_bound.append(conn_free)
                connectors_all.append(conn_bound)

        ## make prvate and force immutable
        self.__connectors = tuple(connectors_all)
        self.__connectors_free = tuple(connectors_free)
        self.__connectors_bound = tuple(connectors_bound)

        # connectors : dict[ConnectorAddress, Connector] = {
            # ...
        # }
        
        # Validate and set topology
        # check_primitive_registry_bijective_to_topology_nodes(children, topology)
        # check_connections_bijective_to_topology_edges(connections, topology)
        
        self.__shape = shape
        self.__metadata = metadata or dict()
    
    # Protected access properties
    @property
    def shape(self) -> Optional[BoundedTransformableShape]:
        return self.__shape
    
    @property
    def metadata(self) -> dict[Hashable, Any]:
        return self.__metadata # DEV: is it possible (or worth it) to prevent the object handed back from being edited (e.g. a View?)

    # Managing Connections
    def connector(self, conn_addr : ConnectorAddress) -> Connector:
        return self.__connectors[conn_addr]

    @property
    def connectors_free(self) -> Collection[Connector]:
        '''
        Connectors whose have not yet been assigned a neighbor
        '''
        return self.__connectors_free
        
    @property
    def connectors_bound(self) -> Collection[Connector]:
        '''
        Connectors (originating from children as they must) which are
        bound and whose neighbor is also a child of this Composite
        '''
        return self.__connectors_bound

    # cached properties
    ...
        
    # Overriding RigidlyTransformable contracts to apply recursively to children as well
    def _copy_untransformed(self) -> 'Primitive':
        raise NotImplementedError
        
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        # TB DEV: shouldn't even be possible, if truly immutable
        raise NotImplementedError
        
class MutableCompositePrimitive(CompositePrimitive): # DEV: this will behave by far the closest to the current Primitive impl
    '''
    A CompositePrimitive which allows for dynamic modification of its internal structure
    (i.e. adding/removing children and connections at will)
    '''
    def __init__( # DEV: could omit entirely; repeated for documentation purposes, and in case extra init config needs to be addeds
        self,
        children : Optional[Iterable[Primitive]]=None,
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict]=None, 
    ) -> None:
        # Initialize bookkeeping attrs
        self.connections : set[Connection] = set()
        self.children_by_address : dict[PrimitiveAddress, Primitive] = dict()
        
        # Bind subprimitives and set connectivity, if possible
        for subprimitive in children:
            self.attach_child(subprimitive)

        if children is None:
            children = tuple()
        self.children = children
        
        self._shape = shape
        self.metadata = metadata or dict()

    # Managing Connections
    ## N.B: deliberately omitted register_connectors/deregister_connectors to enforce the 
    ## constraint that only Simples can inject/withdraw Connectors from the hierarchy

    def connector(self, conn_addr : ConnectorAddress) -> Connector:
        origin_child : Primitive = self.child(self.connector_origin_address[conn_addr])
        return origin_child.connector(conn_addr)

    @property
    def connectors_bound(self) -> Collection[Connector]:
        '''
        Connectors (originating from children as they must) which are
        bound and whose neighbor is also a child of this Composite
        '''
        ...

    @property
    def connectors_free(self) -> Collection[Connector]:
        '''
        Connectors whose have not yet been assigned a neighbor
        '''
        ...
    
    # Hierarchy management
    def child(self, prim_addr : PrimitiveAddress) -> Primitive:
        return self.children_by_address[prim_addr] # raise KeyError if not present

    ## Attaching new children
    def _pre_attach(self, parent : 'MutableCompositePrimitive') -> None:
        '''Preconditions prior to attempting attachment of this Primitive to a parent'''
        if not isinstance(parent, MutableCompositePrimitive):
            raise TypeError('Only MutableCompositePrimitive can be dynamically made parents of other Primitive instances')
    
    def attach_child(self, child : Primitive) -> PrimitiveHandle:
        '''Register a new child Primitive as existing below this one in the resolution hierarchy'''
        child.parent = self
        
        child_address : PrimitiveAddress = child.address()
        self.children_by_address[child_address] = child
        
        # for conn_addr, conn in child.connectors_by_address.items():
        #     self.connector_is_internal[conn_addr] = False # all new connectors are external by default until their are paired into a connection
        #     self.connector_origin_address[conn_addr] = child_address
    
    def _post_attach(self, parent : 'MutableCompositePrimitive') -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        ...

    ## Detaching extant children
    def _pre_detach(self, parent : 'Primitive') -> None:
        '''Preconditions prior to attempting detachment of this Primitive from a parent'''
        if not isinstance(parent, MutableCompositePrimitive):
            raise TypeError('Only MutableCompositePrimitive can be dynamically made parents of other Prmitive instances')
        
    def detach_child(self, prim_addr : PrimitiveAddress) -> Primitive:
        subprimitive = self.child(prim_addr)
        subprimitive.parent = None
        
        del self.children_by_address[prim_addr]
        # for conn_addr, conn in subprimitive.connectors_by_address.items():
        #     del self.connector_is_internal[conn_addr]
        #     del self.connector_origin_address[conn_addr]
            # TODO: free Connectors at the "other end" of any connections to these Connectors
        
        return subprimitive
    
    def _post_detach(self, parent : 'Primitive') -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
    
    ## Managing connections
    ...

    ## Topology editing
    def set_connectivity_from_topology(
        self,
        topology : nx.Graph,
    ) -> None:
        infer_connections_from_topology(
            topology,
            mapped_connectors=dict(),
            n_iter_max=10*len(topology), # TB TODO: fill in actual llogic fordecisidng this - 10 is a number I made up for now
        )

    # Resolution shift operations
    def expand(self) -> None:
        '''Replace this Primitive with its children, preserving connections and traces'''
        raise NotImplementedError

    def flatten(self) -> None:
        '''Recursively expand until all childless subprimitives are depth 1 below this one'''
        raise NotImplementedError

    def contract(self, parts : Iterable[AbstractSet[PrimitiveHandle]], implicit_parts : bool=True) -> None:
        '''
        Insert a new level of Primitive between this Composite and its children,
        with each part of the provided partition forming a new child Primitive
        
        Behavior of implicit parts (i.e. any not explicitly mentioned in "parts")
        can be specified via the "implicit_parts" argument
        ''' # DEV: eventually, make enum for implicit_parts behavior
        raise NotImplementedError

    def truncate(self) -> None:
        '''
        Replace this MutableComposite with an analogous MutableSimple,
        disconnecting all children from the rest of the hierarchy tree
        '''
        raise NotImplementedError
        
    # Overriding RigidlyTransformable contracts to apply recursively to children as well
    def _copy_untransformed(self) -> 'Primitive':
        raise NotImplementedError

# DEV: chose deliberately to not make this a method of any composite subtype
# to avoid requiring subtypes to have awareness of one another in their impl
def frozen(composite : CompositePrimitive) -> FrozenCompositePrimitive:
    '''
    Return an immutable CompositePrimitive copy of this MutableCompositePrimitive
    '''
    if isinstance(composite, FrozenCompositePrimitive):
        return composite
    
    raise NotImplementedError


# Hashable canonical forms for core components
def canonical_form_shape(primitive : Primitive) -> str:
    '''A canonical string representing this Primitive's shape'''
    return type(primitive.shape).__name__ # TODO: move this into .shape - should be responsibility of individual Shape subclasses

def canonical_form_primitive(primitive : Primitive) -> str: # NOTE: deliberately NOT a property to indicated computing this might be expensive
    '''A canonical representation of a Primitive's core parts; induces a natural equivalence relation on Primitives
    I.e. two Primitives having the same canonical form are to be considered interchangable within a polymer system
    '''
    return f'(connectors={canonical_form_connectors(primitive)})' \
        f'[shape={canonical_form_shape(primitive)}]' \
        # f'<graph_hash={self.canonical_form_topology()}>'