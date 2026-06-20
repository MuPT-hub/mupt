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
    Generator,
    Hashable,
    Iterable,
    Optional,
    Mapping,
    Self,
    TypeVar,
)
type PrimitiveLabel =Hashable
type PrimitiveAddress = Hashable
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)

from copy import deepcopy

from anytree import NodeMixin, findall
import networkx as nx

import numpy as np
from scipy.spatial.transform import RigidTransform

from .connection.connectors import (
    Connector,
    canonical_form_connectors,
)
from .connection.exceptions import (
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
from .connection.types import ConnectorAddress, ConnectorLabel, ConnectorHandle
from connection.management import (
    ConnectorManager,
    ConnectorManagerFrozen,
    ConnectorManagerMutable,
)

from .topology import GraphLayout, canonical_graph_property
from .linking import (
    infer_connections_from_topology,
    flexible_connector_reference,
    check_primitive_registry_bijective_to_topology_nodes,
    check_connections_bijective_to_topology_edges,
    check_connections_compatible_with_primitive_registry,
)

from ..geometry.arraytypes import Array3x3
from ..geometry.shapes import Shaped, BoundedTransformableShape
from ..geometry.transforms.rigid import RigidlyTransformable
from ..mutils.containers import UniqueRegistry, Labelled
from ..chemistry.core import ElementLike, isatom, valence_allowed
from ..roles import PrimitiveRole


# Custom Exceptions
class ImproperHierarchyError(AttributeError):
    '''
    Raised when attempting to use a sybtype of Primtiive in a
    place where it can't be used in a hierarchical representation
    '''
    pass

class ArborescenceError(ImproperHierarchyError):
    '''Raised when trying to use a Root as the child of another Primitive'''
    pass

class IrreducibilityError(ImproperHierarchyError):
    '''Raised when attempting to perform a composite Primitive operation on a simple one'''
    pass

class AtomicityError(IrreducibilityError):
    '''Raised when attempting to perform a composite Primitive operation on a simple one (or vice-versa)'''
    pass

class MissingSubprimitiveError(KeyError):
    '''Raised when a child Primitive expected for a call is not present'''
    pass


# Selection strategies
PrimitiveSelector = Callable[['Primitive'], bool]

def indiscriminate_selector(prim : 'Primitive') -> bool:
    '''
    Selector which always greenlights the passed Primitive no matter what
    Useful for avoiding lamba overhead
    '''
    return True

def select_primitives(
    choices : Iterable['Primitive'],
    criterion : Optional[PrimitiveSelector]=None,
) -> Generator['Primitive', None, None]:
    '''Boilerplate for choosing Primitives out of an iterable by some rule'''
    if criterion is None:
        criterion = indiscriminate_selector

    for prim in choices:
        if criterion(prim):
            yield prim


# Primitive base types
class Primitive(Labelled, Shaped, RigidlyTransformable, NodeMixin):
    '''
    A fundamental, scale-agnostic building block of a molecular system
    '''
    # Attributes
    ## Expected classwide attributes
    DEFAULT_LABEL : ClassVar[PrimitiveLabel]

    # Expected instance attributes
    shape : BoundedTransformableShape
    connections : ConnectorManager
    metadata : dict[Hashable, Any]

    _frozen_connections : bool
    _frozen_hierarchy : bool

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

    ## Mutability flags
    @property
    def frozen_connections(self) -> bool:
        '''Whether or not the hierarchy tree is open to Connector modification'''
        return self._frozen_connections
    
    def _precondition_mutable_connectors(
        self,
        msg : str='Connectors of this Primitive are read-only accessible',
    ) -> None:
        '''Boilerplate for checking if permission exists to modify connectivity of this Primitive'''
        if self.frozen_connections:
            raise PermissionError(msg)
        
    @property
    def frozen_hierarchy(self) -> bool:
        '''Whether or not editing incoming or outgoing nodes of this hierarchy is allowed'''
        return self._frozen_hierarchy  

    def _precondition_mutable_hierarchy(
        self,
        msg : str='Hierarchy of this Primitive is read-only accessible; no new incoming or outgoing relationships allowed',
    ) -> None:
        '''Boilerplate for checking if permission exists to modify hierarchical relationships to this Primitive'''
        if self.frozen_hierarchy:
            raise PermissionError(msg)

    # Geometry
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        '''Apply a rigid transformation to all parts of a Primitive which support it'''
        if isinstance(self.shape, RigidlyTransformable):
            self.shape.rigidly_transform(transformation)
            
        for connector in self.connections.connectors:
            connector.rigidly_transform(transformation)
            
    def _copy_untransformed(self) -> Self:
        return NotImplemented

    # Topology  
    def _freeze_connections_local(self) -> None:
        '''Force Connectors on this Primitive to be immutable and cached (without recursive calls)'''
        self.connections = ConnectorManagerFrozen(vars(self.connections)) # TODO: this handoff need work

    def freeze_connections(self) -> None:
        '''Prevent any connection within the hierarchy from being mutated'''
        self._freeze_connections_local()
        for subprimitive in self.children:
            subprimitive._freeze_connections_local()
        self._frozen_connections = True # don't update flag until recursive call completes

    def _unfreeze_connections_local(self) -> None:
        '''Allow Connectors on this Primitive to be mutated (without recursive calls)'''
        self.connections = ConnectorManagerMutable(vars(self.connections)) # TODO: this handoff need work

    def unfreeze_connections(self) -> None:
        '''Enable mutation of connectivity throughout the hierarchy'''
        self._unfreeze_connections_local()
        for subprimitive in self.ancestors:
            subprimitive._unfreeze_connections_local()
        self._frozen_connections = False # don't update flag until recursive call completes

    ## Adjacency
    def neighbors(self, criterion : Optional[PrimitiveSelector]=None) -> Generator['Primitive', None, None]:
        '''Primitives whose share a Connection with this one'''
        for conn in self.connections.connectors_bound:
            yield from select_primitives(
                conn.holders,
                criterion=criterion,
            )

    # Hierarchy
    def search_hierarchy_by(
        self,
        criterion : PrimitiveSelector,
        halt_when : Optional[PrimitiveSelector]=None,
        to_depth  : Optional[int]=None,
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
            filter_=criterion,
            stop=halt_when,
            maxlevel=to_depth,
            mincount=min_count,
            maxcount=max_count,
        )

    # Depiction
    def __hash__(self) -> int:
        # NOTE: !CRITICAL! this be implemented if Primitives are to be used as nodes in networkx graphs
        raise NotImplementedError

    def __str__(self) -> str:
        # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        # return self.canonical_form() # self.canonical_form_salted()
        raise NotImplementedError
    
    def __repr__(self) -> str:
        raise NotImplementedError # TODO - will likely have to change for subtypes
    
class SupportsChildren(Primitive):
    '''
    Type of Primitive which is allowed to have other Primitives "beneath" it in a hierarchy
    I.e. in a rooted tree, these Primitives are nodes which allow OUTGOING directed edges
    '''
    # Hierarchy
    ## Lookup
    children_by_address : UniqueRegistry[PrimitiveAddress, Primitive]
   
    def child(self, prim_addr : PrimitiveAddress) -> SupportsParents:
        # TODO: provide overload which uses a handle <-> address isomorphism
        return self.children_by_address[prim_addr] # raise KeyError if not present
    
    ## Attachment
    def _pre_attach_children(self, children : Iterable[SupportsParents]) -> None:
        '''Preconditions prior to attempting attachment of this Primitive to a parent'''
        ...
        self._precondition_mutable_hierarchy(
            msg='Hierarchy modification is frozen on this Primitive; cannot attach new outgoing nodes'
        )
    
    def _post_attach_children(self, children : Iterable[SupportsParents]) -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        # TODO: remap connection info
        ...

    def attach_child(self, child : Primitive, label : Optional[PrimitiveLabel]=None) -> PrimitiveHandle:
        '''Register a new child Primitive as existing below this one in the resolution hierarchy'''
        child.parent = self
        
        child_address : PrimitiveAddress = child.address()
        self.children_by_address[child_address] = child

    ## Detachment
    def _pre_detach_children(self, parent : SupportsChildren) -> None:
        '''Preconditions prior to attempting detachment of this Primitive from a parent'''
        self._precondition_mutable_hierarchy(
            msg='Hierarchy modification is frozen on this Primitive; cannot detach extant outgoing node(s)'
        )
    
    def _post_detach(self, parent : SupportsChildren) -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        ...

    def detach_child(self, prim_addr : PrimitiveAddress) -> Primitive:
        subprimitive = self.child(prim_addr)
        subprimitive.parent = None
        
        del self.children_by_address[prim_addr]
        # for conn_addr, conn in subprimitive.connectors_by_address.items():
        #     del self.connector_is_internal[conn_addr]
        #     del self.connector_origin_address[conn_addr]
            # TODO: free Connectors at the "other end" of any connections to these Connectors
        
        return subprimitive

    ## Overriding RigidlyTransformable contracts to apply recursively to children as well
    def _copy_untransformed(self) -> 'Primitive':
        raise NotImplementedError
        
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        raise NotImplementedError

    # Topology
    def set_connectivity_from_topology(
        self,
        topology : nx.Graph,
        criterion : PrimitiveSelector,
    ) -> None:
        '''Form connections from a labelled graph, paying respect to selectivity of Connectors'''
        prim_subselection = select_primitives(self.ancestors, criterion=criterion)
        assert len(prim_subselection) == topology.number_of_nodes()
        assert set(prim_subselection.keys()) == set(topology.nodes) # TODO: make return include labels for indication
        
        infer_connections_from_topology(
            topology,
            mapped_connectors={
                prim_label : set(prim.connections.connectors)
                    for prim_label, prim in prim_subselection.items()
            },
            n_iter_max=10*len(topology), # TB TODO: fill in actual llogic fordecisidng this - 10 is a number I made up for now
        )

    def export_cross_section(self, criterion : PrimitiveSelector) -> nx.Graph:
        '''Generate a graph of a "slice" of a subset of sub-Primitives specified by a criterion'''
        raise NotImplementedError

class SupportsParents(Primitive):
    '''
    Type of Primitive which is allowed to have other Primitives "above" it in a hierarchy
    I.e. in a rooted tree, these Primitives are nodes which allow INCOMING directed edges
    '''
    # Hierarchy 
    ## TB: you might be thinking it would be more natural to have checks on Parents in SupportParent instead
    ## the reason for doing this instead is that setting children always calls `child.parent = new_parent_value` under the hood
    def _pre_attach(self, parent : SupportsChildren) -> None:
        self._precondition_mutable_hierarchy()
        parent._precondition_mutable_hierarchy()

    def _post_attach(self, parent : SupportsChildren) -> None:
        ...

    def _pre_detach(self, parent : SupportsChildren) -> None:
        self._precondition_mutable_hierarchy()
        parent._precondition_mutable_hierarchy()

    def _post_detach(self, parent : SupportsChildren) -> None:
        ...

# Concrete primitive types
## Tree root
class RootPrimitive(SupportsChildren):
    '''
    Base of a hierarchy tree - no Primitives can exist above (i.e. own) this one
    Used to store system-wide metadata, as well as provide hand-off point for interfaces
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'ROOT'

    def __init__(
        self,
        box_vectors : Optional[Array3x3]=None,
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        if box_vectors is None:
            box_vectors = np.zeros(3, 3, dtype=float)
        self.box_vectors = box_vectors

        self.connections = ConnectorManagerMutable()
        self._shape = shape
        self.metadata = metadata or dict()

    # DEV: deliverately excluded public setter for is_frozen; this should never be tampered with externally

    # Managing hierarchy
    ## Explicitly banning parents
    def _pre_attach(self, parent : SupportsChildren) -> None:
        raise ArborescenceError('Cannot make Root of hierarchy the child of another Primitive')

    def _pre_detach(self, parent : SupportsChildren) -> None:
        raise ArborescenceError('Invalid state: Root is somehow the child of another Primitive')
  
## Composites
class CompositePrimitive(SupportsChildren, SupportsParents):
    '''
    Primitive representing intermediate levels of organization in a chemical system;
    In a representation hierarchy, always lives between Roots and Simples
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'COMPOSITE'

    def __init__(
        self,
        children : Optional[Iterable[Primitive]]=None,
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict]=None, 
    ) -> None:
        
        # Bind subprimitives and set connectivity, if possible
        self.children_by_address : dict[PrimitiveAddress, Primitive] = dict()
        if children is None:
            children = tuple()
        self.children = children

        connections : ConnectorManager = ConnectorManagerMutable()
        for subprimitive in children:
            self.attach_child(subprimitive)
            connections.register_connectors(
                free=subprimitive.connections.connectors_free,
                bound=subprimitive.connections.connectors_bound,
            )

        self._shape = shape
        self.metadata = metadata or dict()
        self.connections = connections
    
    # Hierarchy
    ...

    ## Resolution shift operations
    def expand(self) -> None:
        '''Replace this Primitive with its children, preserving connections and traces'''
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    def flatten(self) -> None:
        '''Recursively expand until all childless subprimitives are depth 1 below this one'''
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    def contract(self, parts : Iterable[AbstractSet[PrimitiveHandle]], implicit_parts : bool=True) -> None:
        '''
        Insert a new level of Primitive between this Composite and its children,
        with each part of the provided partition forming a new child Primitive
        
        Behavior of implicit parts (i.e. any not explicitly mentioned in "parts")
        can be specified via the "implicit_parts" argument
        ''' # DEV: eventually, make enum for implicit_parts behavior
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    def truncate(self) -> None:
        '''
        Replace this MutableComposite with an analogous MutableSimple,
        disconnecting all children from the rest of the hierarchy tree
        '''
        self._precondition_mutable_hierarchy()
        raise NotImplementedError
        
    ## Topology
    ...

## Simples
class SimplePrimitive(SupportsParents):
    '''
    A Primitive with no internal structure (i.e. no children, topology, or internal connections)
    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    '''
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'SIMPLE'
    
    def __init__(
        self,
        connections : ConnectorManager,
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        self.connections = connections
        self._shape = shape
        self.metadata = metadata or dict()
    
    # Exposing Connectors
    def inject_connector(
        self,
        connector : Connector,
        label : Optional[ConnectorLabel]=None,
    ) -> ConnectorHandle:
        '''Introduce a new Connector into circulation throught the hierarchy'''
        if label is None:
            label = Connector.DEFAULT_LABEL

        conn_handle = self.connections._register_connector(connector, label=label)
        if self.parent:
            for anc in self.ancestors:
                _ = anc.connections._register_connector(connector, label=label)

        return conn_handle

    def withdraw_connector(self, connector : Connector | ConnectorHandle) -> Connector:
        '''Remove a Connector from all levels of a hierarchy'''
        if not isinstance(connector, Connector):
            connector = self.connections.connector(connector)

        self.connections._remove_connector(connector)
        if self.parent:
            for anc in self.ancestors:
                _ = anc.connections._remove_connector(connector)
        
        return connector

    # Explicit ban on attachment of children (already simple)
    def _pre_attach_children(self, children : Iterable[Primitive]) -> None:
        raise IrreducibilityError('Cannot attach child Primitives to a SimplePrimitive instance')

    def _pre_detach_children(self, children : Iterable[Primitive]) -> None:
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
        connections : ConnectorManager,
        shape : Optional[BoundedTransformableShape]=None,
        metadata : Optional[dict]=None,
    ) -> None:
        if not isatom(element):
            raise TypeError(f'Invalid element type {type(element)}')
        self._element = element
        
        super().__init__(
            connections=connections,
            shape=shape,
            metadata=metadata,
        )

    @property # DEV: no setter implemented; element is immutable after instantiation
    def element(self) -> ElementLike:
        '''The chemical element, ion, or isotope associated with this AtomicPrimitive'''
        return self._element
    
    def check_valence(self) -> None:
        '''Check that element assigned to atomic Primitives and bond orders of Connectors are chemically-compatible'''
        valence : float = self.connections.valence
        if not valence_allowed(
            self.element.number,
            self.element.charge,
            valence,
        ):
            raise ValueError(f'Atomic {self!r} with total valence {valence} incompatible with assigned element {self.element!r}')
    
    def canonical_form(self) -> str:
        return f'{self.element.symbol}{canonical_form_primitive(self)}'


# Hashable canonical forms for core components
def canonical_form_shape(primitive : Primitive) -> str:
    '''A canonical string representing this Primitive's shape'''
    return type(primitive.shape).__name__ # TODO: move this into .shape - should be responsibility of individual Shape subclasses

def canonical_form_primitive(primitive : Primitive) -> str: # NOTE: deliberately NOT a property to indicated computing this might be expensive
    '''A canonical representation of a Primitive's core parts; induces a natural equivalence relation on Primitives
    I.e. two Primitives having the same canonical form are to be considered interchangable within a polymer system
    '''
    return f'(connectors={canonical_form_connectors(primitive.connections.connectors)})' \
        f'[shape={canonical_form_shape(primitive)}]' \
        # f'<graph_hash={self.canonical_form_topology()}>'