'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Any,
    ClassVar,
    Container,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
)
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)
from collections import defaultdict, Counter

from anytree.node import NodeMixin
from anytree.search import findall_by_attr
from networkx import get_edge_attributes

from periodictable.core import Element
from scipy.spatial.transform import RigidTransform
import networkx as nx

from .canonicalize import (
    lex_order_multiset,
    lex_order_multiset_str,
)
from .connection import Connector, ConnectorLabel, ConnectorHandle
from .topology import TopologicalStructure
from .embedding import register_connectors_to_topology

from ..mutils.containers import UniqueRegistry
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import RigidlyTransformable


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
    element : Optional[Element]
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
        element : Optional[Element]=None,
        connectors : Optional[Iterable[Connector]]=None,
        children : Optional[Iterable['Primitive']]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : Optional[dict[Hashable, Any]]=None,
    ) -> None:
        # essential components
        ## external bounded shape
        self._shape = None
        self.shape = shape # call validated Shape setter
        
        ## atomic chemistry (when applicable)
        self._element = None
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
        self._external_connectors_map : dict[ConnectorHandle, tuple[PrimitiveHandle, ConnectorHandle]] = dict()
        
        # additional descriptors
        self.label = type(self).DEFAULT_LABEL if (label is None) else label
        self.metadata = metadata or dict()
        
        
    # Chemical atom and bond properties
    @property
    def element(self) -> Optional[Element]:
        '''The chemical element associated with this Primitive, if it represents an atom'''
        return self._element
    
    @element.setter
    def element(self, new_element: Optional[Element]) -> None:
        if new_element is None:
            self._element = None
            return
        
        if self.children:
            raise AttributeError('Primitive with non-trivial internal structure cannot be made atomic (i.e. cannot have "element" assigned)')
        if not isinstance(new_element, Element):
            raise TypeError(f'Invalid element type {type(new_element)}')
        self._element = new_element # TODO: also permit setting with Ion instances to keep track of formal charges
    
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
    
    
    # Child Primitives
    @property
    def children_by_handle(self) -> UniqueRegistry[PrimitiveHandle, 'Primitive']:
        '''Mapping from unique handles (i.e. (label, index) pairs) to child Primitives'''
        return self._children_by_handle
    ## DEV: deliberately not providing setter for children_by_handle - must use attach_child() and detach_child() methods
    
    ## DEV: override __children_or_empty with values of self._children_by_handle?
    
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
            raise KeyError(f'No child Primitive with handle "{primitive_handle}" bound to Primitive "{self.label}"')
    fetch_subprimitive = fetch_child

    def link_children(
        self,
        child_1_handle : PrimitiveHandle,
        child_2_handle : PrimitiveHandle,
        child_1_connector_handle : Optional[ConnectorHandle]=None,
        child_2_connector_handle : Optional[ConnectorHandle]=None,
        **edge_attrs,
    ) -> None:
        '''
        Add a topology edge between two already-bound child Primitives, 
        optionally specifying which Connectors on that pair of children should be paired with the edge between them
        '''
        if (child_1_connector_handle is not None) and (child_2_connector_handle is not None):
            edge_attrs[self.CONNECTOR_EDGE_ATTR] = {
                child_1_handle : child_1_connector_handle,
                child_2_handle : child_2_connector_handle,
            }
            
        # DEV: just want to raise error if child doesn't exist
        _ = self.fetch_child(child_1_handle)
        _ = self.fetch_child(child_2_handle)
        self.topology.add_edge(child_1_handle, child_2_handle, **edge_attrs)

    def link_children_from(self, pairs : Iterable[Any]) -> None:
        raise NotImplementedError

    ## Attachment (fulfilling NodeMixin contract)
    def _pre_attach(self, parent : 'Primitive') -> None:
        '''Preconditions prior to attempting attachment of this Primitive to a parent'''
        # DEV: insert any preconditions beyond checking parent is self or one of self's children (already done by NodeMixin)
        ...

    def attach_child(
        self,
        subprimitive : 'Primitive',
        label : Optional[PrimitiveLabel]=None,
        neighbor_handles : Optional[Iterable[PrimitiveHandle]]=None,
        external_connector_pairing : Optional[Mapping[ConnectorHandle, ConnectorHandle]]=None,
    ) -> PrimitiveHandle:
        '''Add another Primitive as a child of this one, updating related attributes in accordance'''
        # exapnd mutable defaults
        if external_connector_pairing is None:
            external_connector_pairing = dict()
            
        if neighbor_handles is None:
            neighbor_handles = tuple()
            
        # bind self to child
        subprimitive.parent = self
        subprim_handle = self.children_by_handle.register(subprimitive, label=label)

        # update topology
        ## add new node for child Primitive
        LOGGER.debug(f'Inserting new node with handle "{subprim_handle}" into parent topology')
        if (subprim_handle in self.topology):
            raise KeyError(f'Primitive labelled "{subprim_handle}" already present in neighbor topology')
        self.topology.add_node(subprim_handle)
        
        ## add edges to topology for each sibling Primitive
        for nb_handle in neighbor_handles:
            _ = self.fetch_child(nb_handle) # verify that neighbor is actually present
            nb_edge = (subprim_handle, nb_handle)
            LOGGER.debug(f'Inserting edge {nb_edge} into parent topology')
            self.link_children(*nb_edge)
            ## DEV: don't want to decide which connectors are paired to neighbor here (needs to be evaluated globally)
            ## ... UNLESS the user is sure here which pair go together AND that the neighbor already been registered by this point
            
        # bind external Connectors (as specified) to parent's Connectors
        for own_conn_handle, child_conn_handle in external_connector_pairing.items():
            self.pair_external_connectors_vertically(own_conn_handle, subprimitive.label, child_conn_handle)
            
        return subprim_handle
            
    def _post_attach(self, parent : 'Primitive') -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        LOGGER.debug(f'Bound Primitive "{str(self)}" to parent Primitive "{str(parent)}"')

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
        target_child.parent = None
        
        # remove from topology
        if (target_handle not in self.topology):
            raise KeyError(f'Cannot detach child Primitive with handle "{target_handle}" which is not present in the parent topology')
        self.topology.remove_node(target_handle)
        
        # unbind external connections
        new_ext_conn_handles : dict[ConnectorHandle, tuple[PrimitiveHandle, ConnectorHandle]] = dict()
        for own_conn_handle, (child_handle, child_conn_handle) in self.external_connectors_map.items():
            # DEV: opted to re-make dict with rejections, rather than checking for names and deleting
            # matching entries subsequently, to make logic more readable and save on double-loop
            if child_handle != target_handle:
                new_ext_conn_handles[own_conn_handle] = (child_handle, child_conn_handle)
                continue
            # NOTE: implicitly unbind by not transferring child entries over to copied dict
            LOGGER.warning(f'Unbound counterpart of external connection "{own_conn_handle}" from detached child Primitive "{child_handle}"; this must be rebound to restore consistency!')
        self._external_connectors_map = new_ext_conn_handles
        
        return target_child
    
    def _post_detach(self, parent : 'Primitive') -> None:
        '''Post-actions to take once attachment is verified and parent is bound'''
        LOGGER.debug(f'Unbound Primitive "{str(self)}" from parent Primitive "{str(parent)}"')
    
    # DEV: also include attach/detach_parent() methods?


    # Connections
    @property
    def connectors(self) -> UniqueRegistry[ConnectorHandle, Connector]:
        '''Mutable collection of all connections this Primitive is able to make, represented by Connector instances'''
        return self._connectors

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
            raise KeyError(f'No Connector with handle "{connector_handle}" bound to Primitive "{self.label}"')
        
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self._connectors)
    
    # DEV: eventually, would be nice to organize the triples in the external connector map into a more relational-database-y form
    @property
    def external_connectors_map(self) -> dict[ConnectorHandle, tuple[PrimitiveHandle, ConnectorHandle]]:
        '''
        Mapping between the connectors found on self and their analogues on child Primitives
        1:1 mapping between these MUST exist for resolution shift operations to be well-defined
        '''
        return self._external_connectors_map
        
    @property
    def external_connectors_by_child(self) -> dict[PrimitiveHandle, dict[ConnectorHandle, ConnectorHandle]]:
        '''
        Mapping from child Primitive handles to a mapping of 
        (child Connector -> own Connector) pairs defined by the external Connector map
        '''
        ext_conn_by_child = defaultdict(dict)
        for own_conn_handle, (child_handle, child_conn_handle) in self.external_connectors_map.items():
            ext_conn_by_child[child_handle][child_conn_handle] = own_conn_handle

        return dict(ext_conn_by_child)
    
    def check_external_connectors_bijectively_mapped(self, recursive : bool=False) -> None:
        '''
        Check whether all external connectors are mapped to a counterpart Connector amongst
        the children of this Primitive AND that no extraneous Connectors are registered
        '''
        if self.is_leaf:
            return # leaf Primitives by definition have no children, so there's no need to mapping of external connections downwards
        
        # Perform cheap counting check
        if (n_bound_connectors := len(self.external_connectors_map)) != self.functionality:
            raise ValueError(f'Cannot bijectively map {n_bound_connectors} mapped external Connectors onto {self.functionality}-functional Primitive labelled "{self.label}"')
        
        # Check that mapped Connector handles agree
        connector_handles = set(self.connectors.keys())
        bound_connector_handles = set(self.external_connectors_map.keys())
        if connector_handles != bound_connector_handles:
            raise KeyError(
                f'Association between external Connections on this Primitive and its children is inconsistent; {len(connector_handles - bound_connector_handles)}'
                f'Connectors as unbound, and {len(bound_connector_handles - connector_handles)} extraneous binding are present'
            )
        
        # Check that all mapped Connectors and their child counterparts actually exist
        for own_connector_handle, (child_handle, child_connector_handle) in self.external_connectors_map.items():
            # DEV: getting attributes here to draw out any errors encountered during fetch, not to actually use connectors anywhere
            _own_conn = self.fetch_connector(own_connector_handle)
            child = self.fetch_child(child_handle)
            _child_conn = child.fetch_connector(child_connector_handle)

            if recursive:
                child.check_external_connectors_bijectively_mapped(recursive=True)

    def pair_internal_connectors_horizontally(
        self,
        child_1_handle : PrimitiveHandle,
        child_1_connector_handle : ConnectorHandle,
        child_2_handle : PrimitiveHandle,
        child_2_connector_handle : ConnectorHandle
    ) -> None:
        '''
        Associate a pair of Connectors between two adjacent children to the edge joining those children
        '''
        child_1 = self.fetch_child(child_1_handle)
        conn_1 = child_1.fetch_connector(child_1_connector_handle)
        
        child_2 = self.fetch_child(child_2_handle)
        conn_2 = child_2.fetch_connector(child_2_connector_handle)

        self.topology.edges[(child_1_handle, child_2_handle)][self.CONNECTOR_EDGE_ATTR] = {
            child_1_handle : child_1_connector_handle,
            child_2_handle : child_2_connector_handle,
        }

    def pair_external_connectors_vertically(
        self,
        own_connector_handle : ConnectorHandle,
        child_handle : PrimitiveHandle,
        child_connector_handle : ConnectorHandle,
    ) -> None:
        '''
        Bind a Connector on this Primitive to its counterpart on a child Primitive,
        performing necessary intermediate checks to maintain self-consistency
        '''
        # verify that all the objects referenced actually exist and are well-defined
        own_conn = self.fetch_connector(own_connector_handle)
        child = self.fetch_child(child_handle)
        child_conn = child.fetch_connector(child_connector_handle)

        # make association between connectors
        LOGGER.debug(f'Designating Connector "{child_connector_handle}" on Primitive tagged "{child_handle}" as counterpart to external Connector "{own_connector_handle}"')
        self.external_connectors_map[own_connector_handle] = (child_handle, child_connector_handle)
                
    def connector_trace(self, connector_handle : ConnectorHandle) -> list[Connector]: # DEV: eventually, make wrapping type set, once figured out how to hash Connectors losslessly
        '''
        Returns a sequence of Connectors, beginning with the referenced Connector on this Primitives,
        whose n-th term is the Connector corresponding to the referenced Connector n-layers deep into the Primitive hierarchy
        '''
        self.check_external_connectors_bijectively_mapped(recursive=False) # trace operation is ill-defined without this condition
        ext_conn_traces = [self.fetch_connector(connector_handle)]
        if not self.is_leaf:
            # recursively trace downwards - this is the reason for not validating the precondition recursively (duplicates effort done here)
            child_handle, child_connector_handle = self.external_connectors_map[connector_handle]
            child = self.fetch_child(child_handle)
            ext_conn_traces.extend(child.connector_trace(child_connector_handle))
        
        return ext_conn_traces


    # Topology
    def compatible_indiscrete_topology(self) -> TopologicalStructure:
        '''
        An indiscrete (i.e. edgeless) topology over the currently-registered child Primitives 
        Passes all necessary self-consistency checks, though not sufficient ones in general
        '''
        new_topology = TopologicalStructure()
        new_topology.add_nodes_from(self.children_by_handle.keys()) # NOTE: no edges; in not filled in, will be caught downstream by stricter preconditions
        
        return new_topology

    @property
    def topology(self) -> TopologicalStructure:
        '''The connectivity of the immediate children of this Primitive'''
        return self._topology

    @topology.setter
    def topology(self, new_topology : TopologicalStructure) -> None:
        # TODO: initialize discrete topology with number of nodes equal to number of children
        if not isinstance(new_topology, TopologicalStructure):
            raise TypeError(f'Invalid topology type {type(new_topology)}')
        self.check_topology_incompatible(new_topology) # raise exception if incompatible

        self._topology = new_topology

    def num_implied_external_connectors_per_child(self, topology: Optional[TopologicalStructure]=None) -> dict[PrimitiveHandle, int]:
        '''Number of external connections on each child Primitive implied by the provided topology'''
        if topology is None:
            topology = self.topology
            
        self.check_topology_incompatible(topology)
        num_ext_conn_implied = dict()
        for handle, subprimitive in self.children_by_handle.items():
            min_degree : int = self.topology.degree[handle]
            n_excess = subprimitive.functionality - min_degree
            if n_excess < 0:
                raise ValueError(f'Cannot embed {subprimitive.functionality}-functional Primitive "{handle}" into {min_degree}-degree node')
            num_ext_conn_implied[handle] = n_excess

        return num_ext_conn_implied
    
    def num_implied_external_connectors(self, topology: Optional[TopologicalStructure]=None) -> int:
        '''
        The total number of connectors on child Primitives that are implied
        to be external at THIS level in the hierachy by the provided topology
        '''
        return sum(self.num_implied_external_connectors_per_child(topology).values())

    def register_connections_to_topology(
            self,
            connector_registration_max_iter: int=3,
            # allow_overwrite_external_connectors : bool=False,
        ) -> dict[PrimitiveHandle, tuple[Connector]]:
        '''
        Attempt to pair up bondable Connectors of child Primitives along edges in the prescribed topology
        
        If successful, will bind the deduced pairs to the edges in self's topology and
        return the child connectors determined to be external at this Primitives level, keyed by their labels.
        PROVISIONALLY, the user must decide WHICH of these perceived external connections maps to which Connector on this Primitive.
        '''
        # Perform necessary checks to ensure this process is well-defined
        topology = self.topology # DEV: holdover from draft where, like many other topology functions here, the topology can be external
        self.check_topology_incompatible(topology)
        if (implied_n_conn_ext := self.num_implied_external_connectors(topology)) != self.functionality:
            raise ValueError(f'Cannot bijectively map {implied_n_conn_ext} external connectors from children onto connections of {self.functionality}-functional Primitive')

        # attempt to pair up Connectors according to topology
        paired_connectors, found_external_connectors = register_connectors_to_topology( # will raise exception is registration is not possible
            labelled_connectors={
                handle : list(subprimitive.connectors.values())
                    for handle, subprimitive in self.children_by_handle.items()
            },
            topology=topology,
            n_iter_max=connector_registration_max_iter,
        )
        # bind results of paring (if successful) to internal topology
        for edge_label, connector_mapping in paired_connectors.items():
            self.topology.edges[edge_label][self.CONNECTOR_EDGE_ATTR] = connector_mapping

        # if allow_overwrite_external_connectors:
            # TODO - implement routine for inferring correct pairing in general

        return found_external_connectors
    
    @property
    def paired_child_connectors(self) -> dict[ # TODO: make this direct attribute, rather than writing to topology edges
            tuple[PrimitiveHandle, PrimitiveHandle], # indexes the edge connecting the pair of children
            dict[PrimitiveHandle, Connector], # maps each child on the edge to the connector on it associated to that edge
        ]:
        '''
        Mapping from paired edges in self's topology to the associated pair of Connectors that make up that edge
        Value for each edge is a dict mapping child Primitive labels to the corresponding Connectors, empty is no such pairing has been made
        '''
        return get_edge_attributes(self.topology, name=self.CONNECTOR_EDGE_ATTR, default=dict())


    # Necessary and sufficient conditions for self-consistency of topology, connectors, and child Primitives
    @property
    def is_simple(self) -> bool:
        '''Whether a Primitive has no internal structure'''
        return self.topology.is_empty and self.isleaf
    
    def _check_children_bijective_to_topology(self, topology: TopologicalStructure) -> None:
        '''
        Check whether a 1:1 correspondence can exist between all child
        Primitives and all elements of the imposed incidence topology
        '''
        if topology.number_of_nodes() != self.num_children: # simpler counting check rules out obviously-incompatible pairs quicker
            raise ValueError(f'Cannot bijectively map {self.num_children} child Primitives onto {topology.number_of_nodes()}-element topology')
        
        node_labels = set(topology.nodes)
        child_handles = set(self.children_by_handle.keys())
        if node_labels != child_handles:
            raise KeyError(
                f'Set underlying topology does not correspond to handles on child Primitives; {len(node_labels - child_handles)} element(s)'\
                f' present without associated children, and {len(child_handles - node_labels)} child Primitive(s) are unrepresented in the topology'
            )

    def _check_functionalities_incompatible_with_topology(self, topology: TopologicalStructure) -> None:
        '''
        Check whether the functionalities of all child Primitives are 
        incompatible with the corresponding nodes in the imposed Topology
        '''
        if self.is_leaf: # self.is_simple
            if self.is_atom:
                ## TODO: include valency checks based on atomic number and formal charge
                ...
            else:
                ## for non-atom leaves, not really clear as of yet whether there even exist universally-required conditions for functionality
                ...
            return # for now, exit early w/o Exception for leaf cases

        for handle, subprimitive in self.children_by_handle.items():
            if subprimitive.functionality < (min_degree := topology.degree[handle]):
                raise ValueError(f'Cannot embed {subprimitive.functionality}-functional Primitive "{handle}" into {min_degree}-degree node')

    def check_topology_incompatible(self, topology: Optional[TopologicalStructure]=None) -> None:
        '''
        Check necessary conditions for a topology to be compatible with self's children and Connectors
        Unlike many other validation checks found here, these conditions must hold EVEN for leaf Primitives
        
        These conditions NOT being met means the topology is definitely incompatible;
        the converse, however, is not true in general i.e. these conditions being
        met does not guarantee compatibility between children and topology
        '''
        if topology is None:
            topology = self.topology
        
        self._check_children_bijective_to_topology(topology) # NOTE: MUST be done before checking functionalities!
        self._check_functionalities_incompatible_with_topology(topology)
                       
    def check_self_consistent(self) -> None:
        '''
        Check sufficient conditions for whether the children of this Primitive, their
        Connectors, and the Topology imposed upon them contain consistent information
        '''
        # 0) Check necessary conditions first
        self.check_topology_incompatible()
        if self.is_simple:
            return # In leaf case, compatibility checks on children don't apply
        
        # Check sufficient conditions if passed
        ## 1) Check external connections are bijectively identified
        self.check_external_connectors_bijectively_mapped(recursive=False)
        
        ## 2) Check all Connectors of children are either explicitly internal or external
        for child_handle, child_conn_map in self.external_connectors_by_child.items():
            num_connectors_external = len(child_conn_map)
            num_connectors_internal = self.topology.degree[child_handle]
            num_connectors_total_expected = num_connectors_internal + num_connectors_external
            child = self.fetch_child(child_handle)

            if (num_connectors_total_expected != child.functionality):
                raise ValueError(f'Connectors on child {child.functionality}-functional Primitive "{child_handle}" not fully accounted for (c.f. {num_connectors_internal} from topology + {num_connectors_external} explicit external connectors)')

        ## 3) Check to see that internal connections of children can be paired up 1:1 along edges in topology
        for edge_labels, child_conn_map in self.paired_child_connectors.items():
            if not child_conn_map:
                raise AttributeError(f'No associated pairing of Connectors assigned for adjacent pair of child Primitives {edge_labels}')
        
    
    # Resolution shift methods
    ## TODO: make out-of-place versions of these methods (via optional_in_place for a start)?
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
        
        # # generate quotient topology with all target_labels merged into single node
        # partition : dict[PrimitiveLabel, set[PrimitiveLabel]] = dict()
        # relabelling : dict[frozenset[PrimitiveLabel], PrimitiveLabel] = dict()
        # ## wrap together target nodes in quotient
        # partition['contracted'] = target_labels
        # relabelling[frozenset(target_labels)] = master_label

        # ## register all other nodes (still need to be preserved faithfully)
        # for untouched_label in (self.unique_child_labels - target_labels):
        #     label_set = frozenset({untouched_label})
        #     partition[untouched_label] = label_set
        #     relabelling[label_set] = untouched_label

        # new_topology = nx.relabel_nodes(
        #     nx.quotient_graph(
        #         self.topology,
        #         partition=partition,
        #         relabel=False,
        #         create_using=TopologicalStructure,
        #     ),
        #     mapping=relabelling,
        # )
        # nx.draw(new_topology, with_labels=True)
        
        # # generate new, intermediate Primitive, whose topology contains all INTERIOR node + edges of the selected subset
        # intermediate_primitive = Primitive(
        #     shape=new_shape,
        #     label=master_label,
        # )
        # ## Duplicate and hierarchically register all spanning edges
        # neighbor_labels = set()
        # for spanning_edge in nx.edge_boundary(self.topology, target_labels):
        #     prim_label_inner, prim_label_outer = spanning_edge
        #     try:  # TODO: make reference order-agnostic to avoid this terribleness
        #         assoc_conns = self.paired_child_connectors[prim_label_inner, prim_label_outer]
        #     except KeyError:
        #         assoc_conns = self.paired_child_connectors[prim_label_outer, prim_label_inner]
            
        #     # transfer copies of internal connection to intermediate
        #     conn_inner = assoc_conns[prim_label_inner]
        #     conn_inner_clone = conn_inner.copy()
        #     new_conn_id = id(conn_inner_clone)
        #     intermediate_primitive.connectors.register(conn_inner_clone) # TODO: copy handle
        #     # TODO: include new master label into linkables of cloned connectors
            
        #     # TODO: bind any external connectors within the selected subset vertically
        #     conn_outer = assoc_conns[prim_label_outer]
        #     neighbor_labels.add(prim_label_outer)  # TODO: DEV: pass these directly into attach_child(), once complete support for these operations is implemented
        # print(intermediate_primitive.connectors, intermediate_primitive.functionality)

        # ## rebind children in selection to intermediate
        # for child_label in target_labels:
        #     intermediate_primitive.attach_child(
        #         self.detach_child(child_label),
        #     )
        # for (child_1_label, child_2_label, data) in self.topology.subgraph(target_labels).edges(data=True):
        #     intermediate_primitive.link_children(
        #         child_1_label,
        #         child_2_label,
        #         **data, # transfer pairing info and any edge metadata, if present
        #     )

        # ## update topology at this level with new intermediate
        # self.attach_child(
        #     intermediate_primitive,
        #     # neighbor_labels=neighbor_labels,
        #     # external_connector_pairing={conn_outer.label : conn_inner.label},
        # )
        # # TODO: label quotiented component to master master label
        # self.topology = new_topology # validate post-binding

    def expand(
        self,
        target_handle : PrimitiveHandle,
    ) -> None:
        '''
        Replace a child Primitive (identified by its label) with its internal topology
        Loosely corresponds to expanding the single node representing the target in self's topology
        by its own underlying topology. Inverse to contraction
        '''
        child_primitive = self.fetch_child(target_handle)
        if child_primitive.is_simple:
            return # cannot expand leaf Primitives any further
        
        # self.check_self_consistent()
        raise NotImplementedError

    def flatten(self) -> None:
        '''Flatten hierarchy under this Primitive, so that the entire tree has depth 1'''
        raise NotImplementedError
    
    
    # Geometry (info about Shape and transformations)
    @property
    def shape(self) -> Optional[BoundedShape]:
        '''The external shape of this Primitive'''
        return self._shape
    
    @shape.setter
    def shape(self, new_shape : Optional[BoundedShape]) -> None:
        '''Set the external shape of this Primitive'''
        if new_shape is None:
            self._shape = None
            return
        
        if not isinstance(new_shape, BoundedShape):
            raise TypeError(f'Primitive shape must be either NoneType or BoundedShape instance, not object of type {type(new_shape.__name__)}')

        new_shape_clone = new_shape.copy() # NOTE: make copy to avoid mutating original (per Principle of Least Astonishment)
        if self._shape is not None:
            new_shape_clone.cumulative_transformation = self._shape.cumulative_transformation # transfer translation history BEFORE overwriting
        
        self._shape = new_shape_clone
            
    ## applying rigid transformations (fulfilling RigidlyTransformable contracts)
    def _copy_untransformed(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        clone_primitive = self.__class__(
            shape=(None if self.shape is None else self.shape.copy()),
            element=self.element,
            # NOTE: connectors and children transferred verbatim below - no need to set in init here
            connectors=None, 
            children=None,
            label=self.label,
            metadata={key : value for key, value in self.metadata.items()},
        )
        
        clone_primitive._connectors = self._connectors.copy(value_copy_method=Connector.copy) 
        for subprimitive in self.children: # recursively copy children
            clone_primitive.attach_child(subprimitive._copy_untransformed())
        
        clone_primitive._external_connectors_map = {
            conn_handle : (child_handle, child_conn_handle)
                for conn_handle, (child_handle, child_conn_handle) in self.external_connectors_map.items()
        }
        clone_primitive.topology = TopologicalStructure(self._topology) # DEV: use validated or "private" attr to set?
    
        return clone_primitive
    
    def _rigidly_transform(self, transform : RigidTransform) -> None: 
        '''Apply a rigid transformation to all parts of a Primitive which support it'''
        if isinstance(self.shape, BoundedShape):
            self.shape.rigidly_transform(transform)
        
        for connector in self.connectors.values():
            connector.rigidly_transform(transform)
            
        # propogate transformation down recursively
        for subprimitive in self.children: 
            subprimitive.rigidly_transform(transform)
            
            
    # Representation methods
    ## canonical forms for core components
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
        elem_form : str = self.element.symbol if self.element is not None else str(None) # TODO: move this to external function, eventually
        return f'{elem_form}({self.canonical_form_connectors()})[{self.canonical_form_shape()}]<{self.topology.canonical_form()}>'

    def canonical_form_peppered(self) -> str:
        '''
        Return a canonical string representation of the Primitive with peppered metadata
        Used to distinguish two otherwise-equivalent Primitives, e.g. as needed for graph embedding
        
        Named for the cryptography technique of augmenting a hash by some external, stored data
        (as described in https://en.wikipedia.org/wiki/Pepper_(cryptography))
        '''
        return f'{self.canonical_form()}-{self.label}' #{self.metadata}'

    ## Comparison methods
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

    ## Display methods
    def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        return self.canonical_form_peppered()
    
    def __repr__(self) -> str:
        repr_attr_strs : dict[str, str] = {
            'shape': self.canonical_form_shape(),
            'functionality': str(self.functionality),
            'topology_type': repr(self.topology),
            'label': self.label
        }
        attr_str = ', '.join(
            f'{attr}={value_str}'
                for (attr, value_str) in repr_attr_strs.items()
        )
        
        return f'{self.__class__.__name__}({attr_str})'
