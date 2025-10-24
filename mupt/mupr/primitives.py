'''Fundamental data structures for multiscale molecular representation'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Iterable,
    Optional,
    TypeVar,
    Union,
    overload,
)
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)

from copy import deepcopy
from collections import defaultdict

from anytree.node import NodeMixin
from anytree.search import findall, findall_by_attr
from anytree.render import (
    RenderTree,
    AbstractStyle,
    AsciiStyle,
    ContStyle,
    ContRoundStyle,
    DoubleStyle,
)
import networkx as nx

from scipy.spatial.transform import RigidTransform
from matplotlib.axes import Axes

from .canonicalize import lex_order_multiset_str
from .connection import (
    Connector,
    ConnectorLabel,
    ConnectorHandle,
    ConnectorSelector,
    select_first,
    make_second_resemble_first,
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
from .topology import TopologicalStructure, GraphLayout
from .embedding import infer_connections_from_topology, ConnectorReference, flexible_connector_reference

from ..mutils.containers import UniqueRegistry
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import RigidlyTransformable
from ..chemistry.core import ElementLike, isatom, BOND_ORDER, valence_allowed


class AtomicityError(AttributeError):
    '''Raised when attempting to perform a composite Primitive operation on a simple one (or vice-versa)'''
    pass

class BijectionError(ValueError):
    '''Raised when a pair of objects expected to be in 1-to-1 correspondence are mismatched'''
    pass

class MissingSubprimitiveError(KeyError):
    '''Raised when a child Primitive expected for a call is not present'''
    pass

class Primitive(NodeMixin, RigidlyTransformable):
    '''Represents a fundamental (but not necessarily irreducible) building block of a polymer system in the abstract 
    Note that, by default ALL fields are optional; this is to reflect the fact that use-cases and levels of info provided may vary
    
    For example, one might object that functionality and number of atoms could be derived from the SMILES string and are therefore redundant;
    However, in the case where no chemistry is explicitly provided, it's still perfectly valid to define numbers of atoms present
    E.g. a coarse-grained sticker-and-spacer model
    
    As another example, a 0-functionality primitive is also totally legal (ex. as a complete small molecule in an admixture)
    But comes with the obvious caveat that, in a network, it cannot be incorporated into a larger component
    
    Parameters
    ----------
    shape : Optional[BoundedShape]
        A rigid shape which approximates and abstracts the behavior of the primitive in space
    element : Optional[Union[Element, Ion, Isotope]]
        The chemical element associated with this Primitive, IFF the Primitive represents an atom
    connectors : list[Connector]
        A collection of sites representing bonds to other Primitives
    children : Optional[list[Primitive]], default []
        Other Primitives which are taken to be "contained within" this Primitive

    label : Optional[Hashable]
        A handle for users to identify and distinguish Primitives by
    metadata : dict[Hashable, Any]
        Literally any other information the user may want to bind to this Primitive
    '''
    CONNECTOR_EDGE_ATTR : ClassVar[str] = 'paired_connectors'
    DEFAULT_LABEL : ClassVar[PrimitiveLabel] = 'Prim'
    
    # Initializers
    def __init__(
        self, # DEV: force all args to be KW-only?
        shape : Optional[BoundedShape]=None,
        element : Optional[ElementLike]=None,
        connectors : Optional[Iterable[Connector]]=None,
        children : Optional[Iterable['Primitive']]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        # essential components
        ## external bounded shape
        self._shape = None
        if shape is not None:
            self.shape = shape
        
        ## atomic chemistry (when applicable)
        self._element = None
        if element is not None:
            self.element = element
        
        ## child Primitives
        self._topology = TopologicalStructure()
        # self._topology = self.compatible_indiscrete_topology()
        self._children_by_handle : UniqueRegistry[PrimitiveHandle, Primitive] = UniqueRegistry()
        if children is not None:
            self._children_by_handle.register_from(children)
        
        ## off-body connections
        self._connectors : UniqueRegistry[ConnectorHandle, Connector] = UniqueRegistry()
        if connectors is not None:
            self._connectors.register_from(connectors)
            
        self._internal_connections : set[frozenset[ConnectorReference]] = set()
        self._external_connectors : dict[ConnectorHandle, ConnectorReference] = dict()
        
        # additional descriptors
        self.label = type(self).DEFAULT_LABEL if (label is None) else label
        self.metadata = metadata or dict()
        
        
    # Chemical atom and bond properties
    @property
    def element(self) -> Optional[ElementLike]:
        '''
        The chemical element, ion, or isotope associated with this Primitive
        Setting an element is an aknowledgement that this Primitive represents a single atom
        '''
        return self._element
    
    @element.setter
    def element(self, new_element : ElementLike) -> None:
        if self.children:
            raise AtomicityError('Primitive with non-trivial internal structure cannot be made atomic (i.e. cannot have "element" assigned)')
        if not isatom(new_element):
            raise TypeError(f'Invalid element type {type(new_element)}')
        self._element = new_element
    
    @property
    def is_atom(self) -> bool:
        '''Whether the Primitive at hand represents a single atom'''
        return self.is_leaf and (self.element is not None)

    @property
    def num_atoms(self) -> int:
        '''Number of atomic Primitives collectively present below this Primitive in the hierarchy'''
        return len(findall_by_attr(self, value=True, name='is_atom'))

    @property
    def is_atomizable(self) -> bool:
        '''Whether the Primitive represents an all-atom system'''
        return self.is_atom or all(subprim.is_atomizable for subprim in self.children)

    def check_valence(self) -> None: # DEV: deliberately put this here (i.e. not next to "valence" def) for eventual peelaway when splitting off AtomicPrimitive
        '''Check that element assigned to atomic Primitives and bond orders of Connectors are chemically-compatible'''
        if not self.is_atom:
            return

        if not valence_allowed(self.element.number, self.element.charge, self.valence):
            raise ValueError(f'Atomic {self._repr_brief(include_functionality=True)} with total valence {self.valence} incompatible with assigned element {self.element!r}')
    

    # Connections
    @property
    def connectors(self) -> UniqueRegistry[ConnectorHandle, Connector]:
        '''Mutable collection of all connections this Primitive is able to make, represented by Connector instances'''
        return self._connectors
    
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self._connectors)

    @property
    def valence(self) -> int:
        '''Electronic valence of the Primitive, i.e. the total bond order of all external-facing Connectors on this Primitive'''
        total_bond_order : float = sum(
            BOND_ORDER.get(conn.bondtype, 0.0)
                for conn in self._connectors.values()
        )
        return round(total_bond_order)
    chemical_valence = electronic_valence = valence # aliases for convenience

    def register_connector(
        self,
        new_connector : Connector,
        label : Optional[ConnectorLabel]=None,
    ) -> tuple[ConnectorHandle, int]:
        '''
        Register a new Connector to this Primitive by the passed label, or if None is provided, the label on the Connector instance
        Generated a unique handle and binds the Connector to that handle, then returns the handle bound
        '''
        if not isinstance(new_connector, Connector):
            raise TypeError(f'Cannot interpret object of type {type(new_connector)} as Connector')
        # TODO: bind new connectors externally to parent (propagate recursively up tree?)
        return self._connectors.register(new_connector, label=label)

    def register_connectors_from(self, new_connectors : Iterable[Connector]) -> None:
        '''Register multiple Connectors to this Primitive from an iterable'''
        self._connectors.register_from(new_connectors)
        
    def connector_exists(self, connector_handle : ConnectorHandle) -> bool:
        '''Verify that a referenced Connector is actually bound to this Primitive'''
        return connector_handle in self._connectors
    
    def fetch_connector(self, connector_handle : ConnectorHandle) -> Connector:
        '''Fetch a Connector with a given handle from bound Connectors'''
        try:
            return self._connectors[connector_handle]
        except KeyError:
            raise MissingConnectorError(f'No Connector with handle "{connector_handle}" bound to {self._repr_brief()}')
        
    @overload
    def fetch_connector_on_child(self, primitive_handle : ConnectorReference) -> Connector:
        ...

    @overload
    def fetch_connector_on_child(self, primitive_handle : PrimitiveHandle, connector_handle : ConnectorHandle) -> Connector:
        ...

    def fetch_connector_on_child(
        self,
        primitive_handle : Union[ConnectorReference, PrimitiveHandle],
        connector_handle : Optional[ConnectorHandle]=None,
    ) -> Connector:
        '''
        Fetch a Connector with a given handle from a given child Primitive, in the
        process verifying that both the referenced child Primitive and Connector exist
        '''
        conn_ref = flexible_connector_reference(primitive_handle, connector_handle)
        return self.fetch_child(conn_ref.primitive_handle).fetch_connector(conn_ref.connector_handle)
        
    @property
    def all_connectors_on_children(self) -> frozenset[ConnectorReference]:
        '''Unordered collection of all Connectors on children, associated as ConnectorReference instances'''
        return frozenset({
            ConnectorReference(child_handle, conn_handle)
                for child_handle, child in self.children_by_handle.items()
                    for conn_handle, conn in child.connectors.items()
        })
    
    ## Internal "between-child" connections
    @property
    def internal_connections(self) -> set[frozenset[ConnectorReference]]:
        '''
        Collections of all connected pairs of child Connections, identified by the handle
        of the child they're attached to and the Connector handle on that child
        
        Each entry corresponds 1-to-1 with an edge in the topology
        '''
        return self._internal_connections 
    
    @property
    def internal_connections_by_pairs(self) -> dict[frozenset[PrimitiveHandle], frozenset[ConnectorReference]]:
        '''Map from unordered pairs of child Primitive handles to unordered pairs of Connector references between that pair of children'''
        # NOTE: prevention of more than one pair between given children enforces "no-multigraph" requirement
        return { 
            frozenset(conn_ref.primitive_handle for conn_ref in connected_pair) : connected_pair
                for connected_pair in self._internal_connections
        }
        
    @property
    def num_internal_connections(self) -> int:
        '''Number of internal connections (i.e. bonded pairs of Connectors) between child Primitives'''
        return len(self._internal_connections)
    
    @property
    def num_internal_connectors(self) -> int: # DEV: this is potentially confusing/easily mixed up w/ "num_internal_connections" - revisit naming
        '''
        Number of Connectors bound up in internal connections - equal to twice the number of internal connections
        '''
        return 2*self.num_internal_connections
        
    def internal_connections_on_child(self, child_handle : PrimitiveHandle) -> dict[ConnectorHandle, ConnectorReference]:
        '''
        Fetch all referenced siblings which are registered as internally-connected to the given child Primitive
        Returns as dict keyed by the connector handles on the target child whose values are the corresponding ConnectorReference on the sibling
        '''
        paired_connectors = dict()
        for (conn_ref1, conn_ref2) in self._internal_connections:
            if conn_ref1.primitive_handle == child_handle:
                paired_connectors[conn_ref1.connector_handle] = conn_ref2
            elif conn_ref2.primitive_handle == child_handle:
                paired_connectors[conn_ref2.connector_handle] = conn_ref1

        return paired_connectors
    
    def internal_connection_between(
        self,
        from_child_handle : PrimitiveHandle,
        to_child_handle : PrimitiveHandle,
    ) -> tuple[ConnectorHandle, ConnectorHandle]:
        '''
        Fetch the ORDERED pair of Connectors making up an internal connection between two given child Primitives
        i.e. reversing the order of input Primitive labels correspondingly reversed the order of output Connectors handles
        
        Returns a tuple of the form (from_child's connector handle, to_child's connector handle)
        '''
        for from_child_conn_handle, conn_ref in self.internal_connections_on_child(from_child_handle).items():
            if conn_ref.primitive_handle == to_child_handle:
                return (from_child_conn_handle, conn_ref.connector_handle)
        else:
            return tuple()
    
    def num_internal_connections_on_child(self, child_handle : PrimitiveHandle) -> int:
        '''Number of internal connections the given child Primitive has made with its siblings'''
        return len(self.internal_connections_on_child(child_handle))
    
    def neighbor_handles(self, child_handle : PrimitiveHandle) -> set[PrimitiveHandle]:
        '''Set of handles of all sibling child Primitives directly connected to the given child Primitive'''
        return set(
            conn_ref.primitive_handle
                for conn_ref in self.internal_connections_on_child(child_handle).values()
        )
            
    def check_internally_connectable(self, conn_ref1 : ConnectorReference, conn_ref2 : ConnectorReference) -> None:
        '''
        Check that a pair of internally-connected Connectors are:
        * attached to distinct children, both of which exist
        * exist themselves on their respective children
        * are bondable with each other
        '''
        if conn_ref1.primitive_handle == conn_ref2.primitive_handle:
            raise IncompatibleConnectorError(f'Child primitive "{conn_ref1.primitive_handle}" cannot be connected to itself')
        
        # performs necessary existence checks for children and their Connectors while fetching
        conn1 = self.fetch_connector_on_child(conn_ref1)
        conn2 = self.fetch_connector_on_child(conn_ref2)

        if not Connector.bondable_with(conn1, conn2):
            raise IncompatibleConnectorError(
                f'Connector {conn_ref1.connector_handle} on Primitive {conn_ref1.primitive_handle} is not bondable with Connector {conn_ref2.connector_handle} on Primitive {conn_ref2.primitive_handle}'
            )
            
    def pair_connectors_internally(self, conn_ref1 : ConnectorReference, conn_ref2 : ConnectorReference) -> None:
        '''
        Associate a pair of Connectors between two adjacent children to the edge joining those children
        '''
        conn_refs = (conn_ref1, conn_ref2)
        self.check_internally_connectable(*conn_refs)
        for conn_ref in conn_refs:
            own_conn = self.unbind_external_connector(
                connector_handle=self.external_connectors_on_child(conn_ref.primitive_handle)[conn_ref.connector_handle]
            ) # DEV: worth returning these now-unbound instances?
        self._internal_connections.add(frozenset(conn_refs))
        LOGGER.debug(f'Paired {conn_ref1!s} with {conn_ref2!s}')
        
    ## External "off-body" connections
    @property
    def external_connectors(self) -> dict[ConnectorHandle, ConnectorReference]:
        '''Mapping between the Connector handles found on self and their analogues on child Primitives'''
        return self._external_connectors
        
    @property # TODO: find way to cache this (requires some guarantee of immutability of children)
    def external_connectors_by_children(self) -> dict[PrimitiveHandle, dict[ConnectorHandle, ConnectorHandle]]:
        '''
        Mapping from child Primitive handles to (child Connector, own Connector) handle pairs defined by the external Connector map
        '''
        ext_conn_by_child = defaultdict(dict)
        for own_conn_handle, child_conn_ref in self.external_connectors.items():
            ext_conn_by_child[child_conn_ref.primitive_handle][child_conn_ref.connector_handle] = own_conn_handle

        return dict(ext_conn_by_child)
    
    def external_connectors_on_child(self, child_handle : PrimitiveHandle) -> dict[ConnectorHandle, ConnectorHandle]:
        '''
        Mapping between Connector handles on a given child and the corresponding Connector handles on self, if that connection is external
        '''
        return self.external_connectors_by_children.get(child_handle, dict())
    
    def num_external_connectors_on_child(self, child_handle : PrimitiveHandle) -> int:
        '''Number of external connections a given child Primitive has mirrored by its parent (self)'''
        return len(self.external_connectors_on_child(child_handle))
    
    def bind_external_connector(
        self,
        child_handle : PrimitiveHandle,
        child_connector_handle : ConnectorHandle,
        label : Optional[ConnectorLabel]=None,
    ) -> ConnectorHandle:
        '''
        Mirror an external connector on one of self's children to self
        '''
        conn = self.fetch_connector_on_child(child_handle, child_connector_handle)
        conn_counterpart = conn.copy()
        own_conn_handle = self.register_connector(conn_counterpart, label=label)
        self._external_connectors[own_conn_handle] = ConnectorReference(
            primitive_handle=child_handle,
            connector_handle=child_connector_handle,
        )
        LOGGER.debug(f'Added Connector "{own_conn_handle}" as counterpart to Connector "{child_connector_handle}" on child Primitive tagged "{child_handle}"')
        
        return own_conn_handle
    
    def unbind_external_connector(self, connector_handle : ConnectorHandle) -> Connector:
        '''
        Remove an external connector from self, leaving the corresponding Connector on the child Primitive intact
        Returns the now-unbound connector instance
        '''
        _ = self.fetch_connector(connector_handle) # verify existence
        own_conn = self._connectors.deregister(connector_handle)
        
        if connector_handle not in self._external_connectors:
            raise UnboundConnectorError(f'Connector "{connector_handle}" bound to {self._repr_brief()} exists, but is not associated with the Connector of any child Primitive')
        del self._external_connectors[connector_handle]

        return own_conn
    
    def connector_trace(self, connector_handle : ConnectorHandle) -> list[Connector]: # DEV: eventually, make wrapping type set, once figured out how to hash Connectors losslessly
        '''
        Returns a sequence of Connectors, beginning with the referenced Connector on this Primitives,
        whose n-th term is the Connector corresponding to the referenced Connector n-layers deep into the Primitive hierarchy
        '''
        ext_conn_traces = [self.fetch_connector(connector_handle)]
        if not self.is_leaf:
            # recursively trace downwards - this is the reason for not validating the precondition recursively (duplicates effort done here)
            child_conn_ref : ConnectorReference = self.external_connectors[connector_handle]
            child = self.fetch_child(child_conn_ref.primitive_handle)
            ext_conn_traces.extend(child.connector_trace(child_conn_ref.connector_handle))

        return ext_conn_traces
    
    ## Consistency checks on Connections
    def check_external_connector_references_valid(self) -> None:
        '''
        Check that the mapped Connectors on self are each represented in the mapping to the associated external Connectors on children
        '''
        if self.is_leaf:
            return # these checks only make sense for Primitives with children
        
        if self.functionality != len(self.external_connectors):
            raise BijectionError(f'{self._repr_brief(include_functionality=True)} only has {len(self.external_connectors)} registered external Connectors')
        
        own_conn_handles = set(self.connectors.keys())
        mapped_conn_handles = set(self.external_connectors.keys())
        if own_conn_handles != mapped_conn_handles:
            raise UnboundConnectorError(
                f'Connector mapping on {self._repr_brief()} is inconsistent; {len(own_conn_handles - mapped_conn_handles)} Connector(s) have no '\
                f'associated Connectors among children, and {len(mapped_conn_handles - own_conn_handles)} mapped Connector(s) are not registered to the Primitive'
            )
            
    def check_internal_connection_references_valid(self) -> None:
        '''
        Check that each pair of internal connections references a pair of Connectors which exist on their respective child Primitive
        '''
        for conn_refs in self.internal_connections:
            self.check_internally_connectable(*conn_refs)

    def check_connector_balance(self) -> None:
        '''
        Check whether total number of Connectors on children matches the number of
        Connectors bound in internal OR external Connectors on the parent (self)
        '''
        num_total_child_connectors = sum(child.functionality for child in self.children)
        if num_total_child_connectors != (self.num_internal_connectors + len(self.external_connectors)):
            raise BijectionError(
                f'{self._repr_brief()} has Connectors unaccounted for; {num_total_child_connectors} total'\
                f'vs {self.num_internal_connectors} internal + {len(self.external_connectors)} external Connectors'
            )

    def check_connectors(self) -> None:
        '''
        Check that all Connectors on children are properly accounted for by either
        external Connectors or internal Connector pairs (connections) on self
        '''
        self.check_connector_balance() # perform quick counting check to rule out necessarily-impossible cases
        self.check_external_connector_references_valid()
        self.check_internal_connection_references_valid()
        
    def check_child_referenced_faithfully(self, primitive_handle : PrimitiveHandle) -> None:
        '''
        Check that a given child Primitive and the Connectors on it are 
        faithfully referenced in the internal topology and connector registries
        '''
        # DEV: looping over all children and calling this check is a less-efficient way of checking self-
        # consistency than the global check_connectors/check_topology_consistent methods provided elsewhere
        
        # 0) check child exists
        child = self.fetch_child(primitive_handle)
        
        # 1) check registries
        num_child_connectors = child.functionality
        num_internal_on_child = self.num_internal_connections_on_child(primitive_handle)
        num_external_on_child = self.num_external_connectors_on_child(primitive_handle)
        
        if num_child_connectors != (num_external_on_child + num_internal_on_child):
            raise BijectionError(
                f'Connectors on child {child._repr_brief(include_functionality=True, label_to_use=primitive_handle)} not fully accounted for '\
                f'(c.f. {num_child_connectors} total vs {num_internal_on_child} internal + {num_external_on_child} external Connectors)'
            )

        # 2) check topology
        if (child_degree := self.topology.degree[primitive_handle]) != num_internal_on_child:
            raise BijectionError(
                f'Primitive "{primitive_handle}" has {num_internal_on_child} registered internal connections versus {child_degree} internal connections suggested by topology'
            )


    # Child Primitives
    ## DEV: override __children_or_empty with values of self._children_by_handle?
    @property
    def children_by_handle(self) -> UniqueRegistry[PrimitiveHandle, 'Primitive']:
        '''
        Mapping from unique handles (i.e. (label, index) pairs) to child Primitives
        Mapping cannot be set directly; to do so, use protected attach_child() and detach_child() methods
        '''
        return self._children_by_handle
    
    @property
    def has_children(self) -> bool:
        '''
        Whether this Primitive contains any sub-Primitives below it
        See also Primitive.is_simple
        '''
        return bool(self.children)
    
    @property
    def num_children(self) -> int:
        '''Number of sub-Primitives this Primitive contains'''
        # return len(self.children)
        return len(self._children_by_handle)
    
    @property
    def unique_child_labels(self) -> set[PrimitiveLabel]: # NOTE: this type annotation SHOULD be from PrimitiveLabel (NOT PrimitiveHandle!)
        '''Set of all unique labels assigned to child Primitives'''
        return set(self.children_by_handle.by_labels.keys())
    
    def child_exists(self, primitive_handle : PrimitiveHandle) -> bool:
        '''Verify that a referenced child Primitive is actually bound to this Primitive'''
        return primitive_handle in self.children_by_handle

    def fetch_child(self, primitive_handle : PrimitiveHandle) -> 'Primitive':
        '''Fetch a Primitive with a given handle from bound child Primitives'''
        try:
            return self.children_by_handle[primitive_handle]
        except KeyError:
            raise MissingSubprimitiveError(f'No child Primitive with handle "{primitive_handle}" bound to {self._repr_brief()}')
    fetch_subprimitive = fetch_child

    ## Attachment (fulfilling NodeMixin contract)
    def _pre_attach(self, parent : 'Primitive') -> None:
        '''Preconditions prior to attempting attachment of this Primitive to a parent'''
        # DEV: insert any preconditions beyond checking parent is self or one of self's children (already done by NodeMixin)
        ...

    def attach_child(
        self,
        subprimitive : 'Primitive',
        label : Optional[PrimitiveLabel]=None,
        neighbor_connections : Optional[dict[ConnectorHandle, tuple[PrimitiveHandle, ConnectorHandle]]]=None,
    ) -> PrimitiveHandle:
        '''
        Add another Primitive as a child of this one in a self-consistent manner
        
        Can optionally supply a mapping from Connectors on the new child
        to neighbors and corresponding bonded Connectors on those neighbors, 
        if those neighbors are known to already be children of this Primitive
        '''
        if neighbor_connections is None:
            neighbor_connections = dict()

        # bind child to self
        subprimitive.parent = self
        subprim_handle = self.children_by_handle.register(subprimitive, label=label)
        
        # node addition is idempotent, if already present - DEV: is there ever a case where we'd NOT want it to be present?
        self.topology.add_node(subprim_handle) # TODO: move to spearate method and add debug LOGGER output
        
        # register connections - NOTE: order matters here! need to insert all connections, then pair up the internal ones
        for conn_handle in subprimitive.connectors: #subprimitive.connectors.keys():
            self.bind_external_connector(subprim_handle, conn_handle)
            
        for subprim_conn_handle, (nb_handle, nb_conn_handle) in neighbor_connections.items():
            self.connect_children(subprim_handle, subprim_conn_handle, nb_handle, nb_conn_handle)
        LOGGER.info(f'Attached child Primitive "{subprim_handle}" to parent Primitive "{self._repr_brief()}"')

        return subprim_handle
            
    def _post_attach(self, parent : 'Primitive') -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        LOGGER.debug(f'Primitive {parent._repr_brief()} assigned as parent of Primitive {self._repr_brief()}')

    def attach_children_from(
        self,
        *children : Iterable[
            Union[
                'Primitive',
                tuple[
                    'Primitive',
                    Optional[PrimitiveLabel],
                    Optional[dict[ConnectorHandle, tuple[PrimitiveHandle, ConnectorHandle]]]
                ]
            ]
        ],
    ) -> list[PrimitiveHandle]:
        '''
        Attach a sequence of children to this Primitive, returning a list of the handles assigned in the order the Primitives appears
        
        Elements in the iterable can either be Primitive instances (in which case default label and no neighbor connections are assumed)
        or 3-tuples of (Primitive, label, neighbor_connections) as specified in the signature for attach_child()
        '''
        handles = []
        for child in children:
            if isinstance(child, Primitive):
                handle = self.attach_child(child)
            else:
                handle = self.attach_child(*child)
            handles.append(handle)
        return handles

    ## Detachment (fulfilling NodeMixin contract)
    def _pre_detach(self, parent : 'Primitive') -> None:
        '''Preconditions prior to attempting detachment of this Primitive from a parent'''
        # DEV: insert any preconditions from detachment
        ...

    def detach_child(
        self,
        target_handle : PrimitiveHandle,
    ) -> 'Primitive':
        '''Remove a child Primitive from this one, update topology and Connectors, and return the excised child Primitive'''
        target_child = self.fetch_child(target_handle)

        # unbind connections
        ## disconnect target from sibling neighbors internally
        for nb_handle in self.neighbor_handles(target_handle):
            self.disconnect_children(target_handle, nb_handle)
        assert self.num_internal_connections_on_child(target_handle) == 0, f'Failed to disconnect all internal connections on child Primitive "{target_handle}"'

        ## remove external connections on self (corresponding 1:1 with those on target after internal disconnection)
        assert self.num_external_connectors_on_child(target_handle) == target_child.functionality, f'Failed to track all external connections to child Primitive "{target_handle}"'
        for target_conn_handle, own_conn_handle in self.external_connectors_on_child(target_handle).items():
            self.unbind_external_connector(own_conn_handle)
        
        # discard from topology (raises Exception if not present in topology)
        self.topology.remove_node(target_handle) # TODO: move to spearate method and add debug LOGGER output
        
        # deregister child from self
        target_child.parent = None
        del self.children_by_handle[target_handle]
        LOGGER.info(f'Detached child Primitive "{target_handle}" from parent Primitive "{self._repr_brief()}"')
        
        return target_child
    
    def _post_detach(self, parent : 'Primitive') -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        LOGGER.debug(f'Primitive {parent._repr_brief()} disowned former child Primitive {self._repr_brief()}')
    # DEV: also include attach/detach_parent() methods?
    
    ## Internal linkage
    def connect_children(
        self, # TODO: include call signature overload with pair of ConnectorReferences
        child_1_handle : PrimitiveHandle,
        child_1_connector_handle : ConnectorHandle,
        child_2_handle : PrimitiveHandle,
        child_2_connector_handle : ConnectorHandle,
        **edge_attrs,
    ) -> None:
        '''
        Forge a new internal connection between a pair of disconnected child Primitives,
        registering that connection as internal on self and inserting a new edge in the self's topology
        '''
        self.pair_connectors_internally(
            conn_ref1=ConnectorReference(
                primitive_handle=child_1_handle,
                connector_handle=child_1_connector_handle,
            ),
            conn_ref2=ConnectorReference(
                primitive_handle=child_2_handle,
                connector_handle=child_2_connector_handle,
            ),
        )
        self.adjoin_child_nodes(
            child_1_handle,
            child_2_handle,
            **edge_attrs,
        )
            
    # def connect_children_from(self, pairs : Iterable[Any]) -> None:
        # raise NotImplementedError
    
    def disconnect_children(
        self, 
        child_1_handle : PrimitiveHandle,
        child_2_handle : PrimitiveHandle,
    ) -> None:
        '''Disconnect all internal connections between a pair of children, making the associated Connectors external'''
        assert child_1_handle != child_2_handle, 'Cannot disconnect a Primitive from itself'
        internal_conn_ref = self.internal_connections_by_pairs.get(frozenset((child_1_handle, child_2_handle)), None)
        if internal_conn_ref is None:
            LOGGER.warning('No internal connections exist between the given pair of children; nothing to disconnect')
            return
        
        self._internal_connections.remove(internal_conn_ref)
        for conn_ref in internal_conn_ref:
            self.bind_external_connector(
                conn_ref.primitive_handle,
                conn_ref.connector_handle,
            )
    
    ## Search
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
    ## Consistency checks between topology and other internal attributes   
    def check_children_bijective_to_topology_nodes(self, topology : TopologicalStructure) -> None:
        '''
        Verify that a 1:1 correspondence exists between the handles of the child
        Primitives registered to this Primitive and the nodes present in the incidence topology
        '''
        if topology.number_of_nodes() != self.num_children:
            raise BijectionError(f'Cannot bijectively map {self.num_children} child Primitives onto {topology.number_of_nodes()}-element topology')
        
        node_labels = set(topology.nodes)
        child_handles = set(self.children_by_handle.keys())
        if node_labels != child_handles:
            raise BijectionError(
                f'Set underlying topology does not correspond to handles on child Primitives; {len(node_labels - child_handles)} element(s)'\
                f' present without associated children, and {len(child_handles - node_labels)} child Primitive(s) are unrepresented in the topology'
            )
    
    def check_internal_connections_bijective_to_topology_edges(self, topology : TopologicalStructure) -> None:
        '''
        Verify that a 1:1 correspondence exists between the internal connections
        (Connectors paired between sibling child Primitives) and the edges present in the incidence topology
        '''
        if topology.number_of_edges() != self.num_internal_connections:
            raise BijectionError(f'Cannot bijectively map {self.num_internal_connections} internal connections onto {topology.number_of_edges()}-edge topology')
        
        edge_labels = set(frozenset(edge) for edge in topology.edges) # cast to frozenset to remove order-dependence
        internal_conn_pairs = set(self.internal_connections_by_pairs.keys())
        if edge_labels != internal_conn_pairs:
            raise BijectionError(
                f'Incident pairs in associated topology do not correspond to internally-connected pairs of child Primitives;'\
                f'{len(edge_labels - internal_conn_pairs)} edge(s) have no corresponding connection, '\
                f'and {len(internal_conn_pairs - edge_labels)} internal connection(s) are unrepresented in the topology'
            )
    
    def check_topology_compatible(self, topology: Optional[TopologicalStructure]=None) -> None:
        '''
        Check sufficient conditions for a topology (if None is passed, assumed to be the one set on this Primitive)
        to be compatible with self's children and Connectors. These conditions hold true EVEN for leaf Primitives
        '''
        if topology is None:
            topology = self.topology
            
        self.check_children_bijective_to_topology_nodes(topology)
        self.check_internal_connections_bijective_to_topology_edges(topology)
    
    ## Topology access and modification
    @property
    def is_simple(self) -> bool:
        '''Whether a Primitive has no internal structure'''
        return self.topology.is_empty and self.is_leaf
    
    @property
    def topology(self) -> TopologicalStructure:
        '''The connectivity of the immediate children of this Primitive'''
        return self._topology

    @topology.setter
    def topology(self, new_topology : TopologicalStructure) -> None:
        raise PermissionError(f'Direct assignment to topology is prohibited; use {self.__class__.__name__}.set_topology() method instead')
        
    def set_topology(self, new_topology : TopologicalStructure, max_registration_iter : int=100) -> None:
        '''
        Assign a new topology to this Primitive, squashing any prior internal connections and attempting to
        deduce them according to the topology provided (i.e. attempting to pair Connectors along each edge)
        
        Deduction of pairs may require more than the default number of iterations of the registration algorithm to finish converging;
        If registration fails the first time, try increasing max_registration_iter
        '''
        if not isinstance(new_topology, TopologicalStructure):
            raise TypeError(f'Invalid topology type {type(new_topology)}')
        
        self.set_connectivity_from_topology(
            new_topology,
            connector_registration_max_iter=max_registration_iter,
        ) # TODO: attempt to reconcile existing internal connections
        self._topology = new_topology
        
        self.check_self_consistent()

    def compatible_indiscrete_topology(self) -> TopologicalStructure:
        '''
        An indiscrete (i.e. edgeless) topology over the currently-registered child Primitives 
        Passes all necessary self-consistency checks, though not sufficient ones in general
        '''
        new_topology = TopologicalStructure()
        new_topology.add_nodes_from(self.children_by_handle.keys())
        
        return new_topology
        
    def adjoin_child_nodes(
        self,
        child_1_handle : PrimitiveHandle,
        child_2_handle : PrimitiveHandle,
        **edge_attrs,
    ) -> None:
        '''Add an edge between two child Primitives in the topology'''
        # verify that children actually exist
        _ = self.fetch_child(child_1_handle) 
        _ = self.fetch_child(child_2_handle)
        self.topology.add_edge(child_1_handle, child_2_handle, **edge_attrs)
        LOGGER.debug(f'Added edge between child Primitives "{child_1_handle}" and "{child_2_handle}" in topology of {self._repr_brief()}')
       
    def set_connectivity_from_topology(
        self,
        topology : TopologicalStructure,
        connector_registration_max_iter: int=25,
    ) -> None:
        '''
        Set internal connections between pairs of child Primitives according to a provided incidence topology (as a graph)
        Attempts to infer which Connectors are pairable along each edge, and will choose first available pair if multiple options exist
        
        Much coarser and less reliable than individually specifying connections, though more expedient for testing/demos
        '''
        self.check_children_bijective_to_topology_nodes(topology)
        LOGGER.warning('Attempting to infer internal connections automatically from given topology; user should verify the connections assigned make sense!')
        
        internal_connections_inferred : dict[
            frozenset[PrimitiveHandle],
            frozenset[ConnectorReference]
        ] = infer_connections_from_topology( 
            topology=topology,
            mapped_connectors=  {
                handle: subprim.connectors
                    for handle, subprim in self.children_by_handle.items()
            },
            n_iter_max=connector_registration_max_iter,
        )
        for (conn_ref_1, conn_ref_2) in internal_connections_inferred.values():
            self.pair_connectors_internally(conn_ref_1, conn_ref_2)
        self.check_internal_connections_bijective_to_topology_edges(topology) # verify that all edges have been accounted for

    
    # Resolution shift methods
    ## TODO: make out-of-place versions of these methods (via optional_in_place for a start)?
    def check_self_consistent(self) -> None:
        '''
        Check sufficient conditions for whether the children of this Primitive, their
        Connectors, and the Topology imposed upon them contain consistent information
        '''
        if self.is_simple:
            return # In leaf case, compatibility checks on children don't apply
        
        self.check_connectors()
        self.check_topology_compatible()
        # DEV: add other checks (as found necessary) here

    def contract(
        self,
        target_labels : set[PrimitiveHandle],
        master_label : PrimitiveHandle,
        new_shape : Optional[BoundedShape]=None,
    ) -> None:
        '''
        Insert a new level into the hierarchy and group the selected
        child Primitives into a single intermediate at the new level
        
        Inverse to expansion
        '''
        raise NotImplementedError
        # ensure all members of subset are present as children
        # self.check_self_consistent()
        # if master_label in self.unique_child_labels:
        #     raise ValueError(f'Cannot contract child Primitives into new intermediate with label "{master_label}" already present amongst children')

        # if not target_labels.issubset(self.unique_child_labels):
        #     raise ValueError('Child Primitives labels chosen for contraction are not a proper subset of the children actually present')

    @property
    def expandable_children(self) -> set[PrimitiveHandle]:
        '''Set of all children (referenced by handle) which are capable of being expanded, ie. replaced with the hierarchy of their children'''
        return {
            child_handle
                for child_handle, child in self.children_by_handle.items()
                    if not child.is_simple
        }

    def expand(
        self,
        target_handle : PrimitiveHandle,
        connector_selector : ConnectorSelector=make_second_resemble_first,
    ) -> None:
        '''
        Replace a child Primitive (identified by its label) with its internal topology
        Loosely corresponds to expanding the single node representing the target in self's topology
        by its own underlying topology. Inverse to contraction
        '''
        child_primitive = self.fetch_child(target_handle)
        if child_primitive.is_simple:
            return # cannot expand leaf Primitives any further
        
        # 1) determine which connections (with prior handles) to re-establish once target is replaced with its children 
        # MUST be done first, as re-attaching and remapping grandchildren will destroy these handles on the target
        LOGGER.info(f'Expanding child Primitive "{target_handle}" of {self._repr_brief()}, replacing it with its {child_primitive.num_children} children')
        self.check_self_consistent() # necessary to be satisfied for reconnection operations to be well-defined
        
        prior_internal_connections : set[frozenset[ConnectorReference]] = set(child_primitive.internal_connections) 
        prior_neighbor_connections_map : dict[ConnectorReference, ConnectorReference] = {
            child_primitive.external_connectors[child_conn_handle] : nb_conn_ref
                for child_conn_handle, nb_conn_ref in self.internal_connections_on_child(target_handle).items()
        } # NOTE: done as dict to delineate which handles DO need to be remapped (namely keys) vs those which don't (namely those of unchanged neighbors)
        
        # 2) detach target from self, attaching its children ("grandchildren" of self) in its place
        handle_remap : dict[PrimitiveHandle, PrimitiveHandle] = {}
        child_primitive_detached = self.detach_child(target_handle) # detach target from self
        
        ## 2a) compatibilize Connectors by collapsing correspondent pairs from external Connectors into single, representative Connector on each grandchild
        for child_conn_handle, grandchild_conn_ref in child_primitive_detached.external_connectors.items():
            child_conn = child_primitive_detached.fetch_connector(child_conn_handle)
            grandchild = child_primitive_detached.fetch_child(grandchild_conn_ref.primitive_handle)
            grandchild_conn_ref_label, grandchild_conn_ref_index = grandchild_conn_ref.connector_handle
            grandchild_conn = grandchild.connectors.deregister(grandchild_conn_ref.connector_handle) 
            
            ### register modified connector to (what ought to be) the same handle
            new_grandchild_conn_handle = grandchild.register_connector( 
                connector_selector(child_conn, grandchild_conn),
                label=grandchild_conn_ref_label, # use identical label to immediately recover handle from "freed" buffer in internal registry
            ) 
            assert new_grandchild_conn_handle == grandchild_conn_ref.connector_handle, 'Connector handle changed unexpectedly during merging'
            
        ## 2b) reassign grandchildren as direct children of self and discard twitching corpse of target child (we have no further use for it from here on)
        for old_handle, grandchild in child_primitive_detached.children_by_handle.items():
            old_label, old_idx = old_handle
            handle_remap[old_handle] = self.attach_child(grandchild, label=old_label) # attach and map handle - # DEV: worth explicitly detaching grandchildren from target?
            
        # 3) re-map previously-established connections to updated handles
        promised_connections : set[frozenset[ConnectorReference]] = set()
        for prior_internal_conn_ref1, prior_internal_conn_ref2 in prior_internal_connections:
            promised_connections.add(
                frozenset((
                    prior_internal_conn_ref1.with_reassigned_primitive(handle_remap[prior_internal_conn_ref1.primitive_handle]),
                    prior_internal_conn_ref2.with_reassigned_primitive(handle_remap[prior_internal_conn_ref2.primitive_handle]),
                ))
            )
        for prior_external_conn_ref, unaltered_nb_conn_ref in prior_neighbor_connections_map.items():
            promised_connections.add(
                frozenset((
                    prior_external_conn_ref.with_reassigned_primitive(handle_remap[prior_external_conn_ref.primitive_handle]),
                    unaltered_nb_conn_ref,
                ))
            )
        
        # 4) re-establish promised connections, following appropriate relabelling
        for (conn_ref1, conn_ref2) in promised_connections:
            self.connect_children(
                conn_ref1.primitive_handle,
                conn_ref1.connector_handle,
                conn_ref2.primitive_handle,
                conn_ref2.connector_handle,
            )
        self.check_self_consistent() # verify that all parts are consistent once the dust settles

    def expanded(
        self,
        target_handle : PrimitiveHandle,
        connector_selector : ConnectorSelector=make_second_resemble_first,
    ) -> 'Primitive': # DEV: eventually wrap with optional_in_place, once I've sorted how to provide custom copy method in general?
        '''Return a copy of this Primitive with the specified child expanded'''
        clone_primitive = self.copy()
        clone_primitive.expand(
            target_handle,
            connector_selector=connector_selector,
        )
        
        return clone_primitive

    def flatten(self) -> None:
        '''Flatten hierarchy under this Primitive, so that the entire tree has height 1'''
        while (target_handles := self.expandable_children): # DEV: also works for simple Primitives (no casework needed!)
            for child_handle in target_handles:
                self.expand(child_handle)
                
    def flattened(self) -> 'Primitive': # DEV: eventually wrap with optional_in_place, once I've sorted how to provide custom copy method in general?
        '''Return a copy of this Primitive which has been flattened'''
        clone_primitive = self.copy()
        clone_primitive.flatten()
        
        return clone_primitive

    # Geometry (info about Shape and transformations)
    @property
    def shape(self) -> Optional[BoundedShape]:
        '''The external shape of this Primitive'''
        return self._shape
    
    @shape.setter
    def shape(self, new_shape : BoundedShape) -> None:
        '''Set the external shape of this Primitive'''
        if not isinstance(new_shape, BoundedShape):
            raise TypeError(f'Primitive shape must be BoundedShape instance, not object of type {type(new_shape.__name__)}')

        new_shape_clone = new_shape.copy() # NOTE: make copy to avoid mutating original (per Principle of Least Astonishment)
        if self._shape is not None:
            new_shape_clone.cumulative_transformation = self._shape.cumulative_transformation # transfer translation history BEFORE overwriting
        
        self._shape = new_shape_clone
            
    ## applying rigid transformations (fulfilling RigidlyTransformable contracts)
    def _copy_untransformed(self) -> 'Primitive':
        '''Return a new Primitive with the same information and children as this one, but which has no parent'''
        clone_primitive = self.__class__(
            shape=(None if self.shape is None else self.shape.copy()),
            element=self.element,
            # NOTE: connectors and children transferred verbatim below - no need to set in init here
            connectors=None, 
            children=None,
            label=self.label,
            metadata=deepcopy(self.metadata),
        )
        
        # transfer connection info
        clone_primitive._connectors = self._connectors.copy(
            value_copy_method=Connector.copy
        )
        clone_primitive._internal_connections = set(self._internal_connections) # frozensets are immutable, so no need to copy deeper
        clone_primitive._external_connectors = {
            conn_handle : ConnectorReference(
                primitive_handle=conn_ref.primitive_handle,
                connector_handle=conn_ref.connector_handle,
            )
                for conn_handle, conn_ref in self.external_connectors.items()
        }
        
        # transfer children
        clone_primitive._children_by_handle = self.children_by_handle.copy(
            value_copy_method=Primitive._copy_untransformed
        )
        for subprimitive in clone_primitive.children_by_handle.values():
            subprimitive.parent = clone_primitive # needs to be rebound, since bypassing attach_child() to preserve handles
        
        # transfer topology
        clone_primitive._topology = TopologicalStructure(self._topology)
        # DEV: include self-consistency check in here for good measure?
    
        return clone_primitive
    
    def _rigidly_transform(self, transformation : RigidTransform) -> None: 
        '''Apply a rigid transformation to all parts of a Primitive which support it'''
        if isinstance(self.shape, BoundedShape):
            self.shape.rigidly_transform(transformation)
        
        for connector in self.connectors.values():
            connector.rigidly_transform(transformation)
            
        # propogate transformation down recursively
        for subprimitive in self.children: 
            subprimitive.rigidly_transform(transformation)
            
            
    # Comparison methods
    def __hash__(self): 
        '''Hash used to compare Primitives for identity (NOT equivalence)'''
        # return hash(self.canonical_form())
        return hash(self.canonical_form_peppered())
    
    def __eq__(self, other : object) -> bool:
        # DEVNOTE: in order to use equivalent-but-not-identical Primitives as nodes in nx.Graph, __eq__ CANNOT evaluate similarity by hashes
        # DEVNOTE: hashing needs to be stricter than equality, i.e. two Primitives may be distinguishable by hash, but nevertheless equivalent
        '''Check whether two primitives are equivalent (but not necessarily identical)'''
        if not isinstance(other, Primitive):
            raise TypeError(f'Cannot compare Primitive to {type(other)}')

        return self.canonical_form() == other.canonical_form() # NOTE: ignore labels, simply check equivalency up to canonical forms
    
    def coincident_with(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are coincident (i.e. all spatial parts are either equally unassigned or occupy the same space)'''
        raise NotImplementedError
    
    def equivalent_to(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are equivalent (i.e. have interchangeable part which are not necessarily in the same place in space)'''
        raise NotImplementedError
            
            
    # Representation methods
    ## Canonical forms for core components
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

    ## Printing and textual representations
    def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        return self.canonical_form_peppered()
    
    def __repr__(self) -> str:
        repr_attr_strs : dict[str, str] = {
            'shape': self.canonical_form_shape(),
            'functionality': str(self.functionality),
            'topology': repr(self.topology),
            'element' : str(self.element),
            'label': self.label
        }
        attr_str = ', '.join(
            f'{attr}={value_str}'
                for (attr, value_str) in repr_attr_strs.items()
        )
        
        return f'{self.__class__.__name__}({attr_str})'

    def _repr_brief(
        self,
        include_functionality : bool=False,
        label_to_use : Hashable=None,
    ) -> str:
        '''A brief representation of this Primitive, suitable for logging'''
        if label_to_use is None:
            label_to_use = self.label

        repr_brief : str = f'{self.__class__.__name__} "{label_to_use}"'
        if include_functionality:
            repr_brief = f'{self.functionality}-functional {repr_brief}'
    
        return repr_brief
    
    def hierarchy_summary(
        self,
        to_depth : Optional[int]=None,
        render_attr : str='label',
        style : Union[str, AbstractStyle]=ContStyle(),
    ) -> str:
        '''A printable representation of this Primitive and all its descendants in the hierarchy'''
        style_aliases : dict[str, AbstractStyle] = { # TODO: move this to external util
            'cont': ContStyle(),
            'Cont': ContStyle(),
            'continued': ContStyle(),
            'ContStyle': ContStyle(),
            'cont_round' : ContRoundStyle(),
            'ContRound' : ContRoundStyle(),
            'ContRoundStyle' : ContRoundStyle(),
            'ascii' : AsciiStyle(),
            'ASCII' : AsciiStyle(),
            'AsciiStyle': AsciiStyle(),
            'double': DoubleStyle(),
            'Double': DoubleStyle(),
            'DoubleStyle': DoubleStyle(),
        }
        if isinstance(style, str):
            style = style_aliases[style]
        
        return RenderTree(
            self,
            style=style,
            # DEV: revisit "childiter" parameter config?
            maxlevel=to_depth,
        ).by_attr(render_attr)
    
    ## Graph drawing
    def visualize_topology(
        self,
        ax : Optional[Axes]=None,
        layout : GraphLayout=nx.kamada_kawai_layout,
        **draw_kwargs,
    ) -> None:
        '''
        Draw the connectivity of this Primitive's children to the passed Axes
        '''
        self.topology.visualize(ax=ax, layout=layout, **draw_kwargs)
        
    def _hierarchy_tree(
        self,
        root_label : Hashable=None,
        depth : int=0,
    ) -> nx.DiGraph:
        '''Generate directed graph representing the connectivity of this Primitive and all its descendants'''
        if root_label is None:
            root_label = self.label
        root_handle = f'{depth}-{root_label}'
        
        hier_tree = nx.DiGraph()
        hier_tree.add_node(root_handle)
        for child_handle, child in self.children_by_handle.items():
            hier_tree = nx.compose(
                hier_tree,
                child._hierarchy_tree(root_label=child_handle, depth=depth+1),
                # rename=(), # TODO: add mechanism for de-duplifying handles by depth (related to planned expansion acceleration)
            )
            hier_tree.add_edge(root_handle, child_handle)
            
        return hier_tree

    def visualize_hierarchy(
        self,
        ax : Optional[Axes]=None,
        layout : GraphLayout=nx.shell_layout,
        **draw_kwargs
    ) -> None:
        '''
        Draw the hierarchy of this Primitive and all its descendants to the passed Axes
        '''
        hier_tree = self._hierarchy_tree()
        nx.draw(
            hier_tree,
            ax=ax,
            pos=layout(hier_tree),
            with_labels=True,
            **draw_kwargs,
        )