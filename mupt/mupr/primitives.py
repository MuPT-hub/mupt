"""Fundamental data structures for multiscale molecular representation"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import (
    Any,
    AbstractSet,  # covers both set and frozenset
    Callable,
    ClassVar,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Optional,
    Self,
    Type,
    Union,
)

type PrimitiveLabel = Hashable
type PrimitiveAddress = Hashable
type PrimitiveHandle = tuple[PrimitiveLabel, int]  # (label, uniquification index)

from weakref import WeakValueDictionary

from anytree.node import NodeMixin
from anytree.render import RenderTree
from anytree.search import findall
from networkx import Graph, DiGraph, MultiGraph

import numpy as np
from scipy.spatial.transform import RigidTransform

from .connection.connectors import (
    Connector,
    canonical_form_connectors,
)
from .connection.exceptions import (  # noqa: F401
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
from .connection.types import (
    ConnectorAddress,
    ConnectorLabel,
)
from .connection.management import (  # noqa: F401
    ConnectorManager,
    ConnectorManagerFrozen,
    ConnectorManagerMutable,
    connector_address_flexible,
)
from .connection.alignment import (  # noqa: F401
    ConnectorAntialignmentStrategy,
    ConnectorAntialignmentRigid,
)
from .linking import (
    deduce_connections_from_topology,
    assign_connections_from_topology,
    GraphIterRule,
)
from .topology import GraphLayout, canonical_graph_property  # noqa: F401
from ..trees.render import tree_render_style, ConcreteStyle
from ..trees.digraph import anytree_to_networkx

from ..mutils.referencing import Addressed
from ..mutils.containers import Labelled
from ..geometry.arraytypes import Array3x3
from ..geometry.shapes import Shaped, BoundedTransformableShape
from ..geometry.transforms.rigid import RigidlyTransformable
from ..chemistry.core import ElementLike, isatom, valence_allowed


# Custom Exceptions
class ImproperHierarchyError(AttributeError):
    """
    Raised when attempting to use a sybtype of Primtiive in a
    place where it can't be used in a hierarchical representation
    """

    pass


class ArborescenceError(ImproperHierarchyError):
    """Raised when trying to use a Root as the child of another Primitive"""

    pass


class IrreducibilityError(ImproperHierarchyError):
    """
    Raised when attempting to perform a
    composite Primitive operation on a simple one
    """

    pass


class AtomicityError(IrreducibilityError):
    """
    Raised when attempting to perform a composite Primitive
    operation on a simple one (or vice-versa)
    """

    pass


class MissingSubprimitiveError(KeyError):
    """Raised when a child Primitive expected for a call is not present"""

    pass


# Selection strategies
PrimitivePredicate = Callable[["Primitive"], bool]


def indiscriminate_selector(prim: "Primitive") -> bool:
    """
    Selector which always greenlights the passed Primitive no matter what
    Useful for avoiding lamba overhead
    """
    return True


# TODO: add pruned BFS to enforce one-prim-per-branch selection


def select_primitives(
    choices: Iterable["Primitive"],
    predicate: Optional[PrimitivePredicate] = None,
) -> Generator["Primitive", None, None]:
    """Boilerplate for choosing Primitives out of an iterable by some rule"""
    if predicate is None:
        predicate = indiscriminate_selector

    for prim in choices:
        if predicate(prim):
            yield prim


# Primitive base types
class Primitive(
    Addressed,
    # TB DEV: Addressed base is potentially problematic,
    # since all subclasses will have separate registries
    Labelled,
    Shaped,
    RigidlyTransformable,
    NodeMixin,
):
    """A fundamental, scale-agnostic building block of a molecular system"""

    # Attributes
    ## Expected classwide attributes
    DEFAULT_LABEL: ClassVar[PrimitiveLabel]

    # Expected instance attributes
    shape: BoundedTransformableShape
    connections: ConnectorManager
    metadata: dict[Hashable, Any]

    _frozen_connections: bool
    _frozen_hierarchy: bool

    ## Derived properties
    @property
    def label(self) -> PrimitiveLabel:
        """
        A distinguishing label which can be assigned
        by the user for identification purposes
        """
        if "label" in self.metadata:
            return self.metadata["label"]
        return self.DEFAULT_LABEL

    ## Functionality-determining properties
    @property
    def is_simple(self) -> bool:
        """
        Whether Primitives are to be considered indivisible
        from the perspective of the hierarchy
        """
        # DEVNOTE: this is a mechanism to prevent Simples from being the parents of any
        # other Primitive without passing type info backward up the inheritance tree
        return False

    ## Mutability flags
    @property
    def frozen_connections(self) -> bool:
        """Whether or not the hierarchy tree is open to Connector modification"""
        return self._frozen_connections

    def _precondition_mutable_connectors(
        self,
        msg: str = "Connectors of this Primitive are read-only accessible",
    ) -> None:
        """
        Boilerplate for checking if permission exists
        to modify connectivity of this Primitive
        """
        if self.frozen_connections:
            raise AttributeError(msg)

    @property
    def frozen_hierarchy(self) -> bool:
        """Whether editing incoming or outgoing nodes of this hierarchy is allowed"""
        return self._frozen_hierarchy

    def _precondition_mutable_hierarchy(
        self,
        msg: str = (
            "Hierarchy of this Primitive is read-only accessible; "
            "no new incoming or outgoing relationships allowed."
        ),
    ) -> None:
        """
        Boilerplate for checking if permission exists to
        modify hierarchical relationships to this Primitive
        """
        if self.frozen_hierarchy:
            raise AttributeError(msg)

    # Geometry
    def _rigidly_transform(self, transformation: RigidTransform) -> None:
        """Apply a rigid transformation to all parts of a Primitive which support it"""
        if isinstance(self.shape, RigidlyTransformable):
            self.shape.rigidly_transform(transformation)

        for connector in self.connections.connectors:
            connector.rigidly_transform(transformation)

    def _copy_untransformed(self) -> Self:
        # TODO: include extra logic from copying bound "edge" Connectors
        # which need to be re-initialized w/out their prevous neighbor
        raise NotImplementedError

    # Topology
    def _freeze_connections_local(self) -> None:
        """
        Force Connectors on this Primitive to be
        immutable and cached WITHOUT recursive calls
        """
        self.connections = ConnectorManagerFrozen(*self.connections.connectors)

    def _freeze_connections_recursive(self) -> None:
        """
        Prevent any connection within the hierarchy at
        this Primitive and below from being mutated
        """
        self._freeze_connections_local()
        for subprimitive in self.children:
            subprimitive._freeze_connections_recursive()
        self._frozen_connections = (
            True  # don't update flag until recursive call completes
        )

    def freeze_connections(self) -> None:
        """
        Prevent connectivity of this Primitive and any
        others in its hierarchy tree from being mutated
        """
        # TB: from root, since it doesn't make sense to just freeze parts of hierarchy;
        # if one bit is frozen, it causes all others touching it to also freeze
        # also note that the "root" here is not necessarily a
        # RootPrimitive, but rather the topmost Primitive ancestor
        self.root._freeze_connections_recursive()

    def _unfreeze_connections_local(self) -> None:
        """Allow Connectors on this Primitive to be mutated (without recursive calls)"""
        self.connections = ConnectorManagerMutable(*self.connections.connectors)

    def _unfreeze_connections_recursive(self) -> None:
        """
        Enable mutation of connectivity for this Primitive
        and any Primitives below it from being mutated
        """
        self._unfreeze_connections_local()
        for subprimitive in self.children:
            subprimitive._unfreeze_connections_recursive()
        self._frozen_connections = (
            False  # don't update flag until recursive call completes
        )

    def unfreeze_connections(self) -> None:
        """
        Enable mutation of connectivity of this Primitive and
        any others below it in the hierarchy tree
        """
        self.root._unfreeze_connections_recursive()

    # Adjacency
    @property
    def connectors(self) -> Collection[Connector]:
        """Convenience wrapper for accessing ALL connectors managed by this Primitive"""
        # TODO: also provide convenient access to connectors_free and connectors_bound
        return self.connections.connectors

    # DEV: purposely excluded connectors.setter and connectors.deleter;
    # connectors access through this property SHOULd be read-only

    def fetch_connector(self, conn: ConnectorAddress | Connector) -> Connector:
        """Fetch a connector managed by this Priomitive, if it exists"""
        return self.connections.connector(connector_address_flexible(conn))

    def neighbors(
        self,
        predicate: Optional[PrimitivePredicate] = None,
    ) -> Generator["Primitive", None, None]:
        """Primitives whose share a Connection with this one"""
        for conn in self.connections.connectors_bound:
            # TB TODO: figure out how to type this so HoldsConnector
            # "knows" about NodeMixin methods without explicitly mentioning
            # base Primitive type in ..connections

            # may include explicit check for has_holder to avoid errant NoneTypes passed
            neighbor_branch: tuple[Primitive] = conn.neighbor.holder.path
            if self in neighbor_branch:
                # avoid "internal" neighbors (of whom this Primitive is also a parent)
                continue

            yield from select_primitives(
                neighbor_branch,
                predicate=predicate,
            )

    def is_neighbors_with(
        self,
        other: "Primitive",
        predicate: Optional[PrimitivePredicate] = None,
    ) -> bool:
        """
        Whether this Primitive is a neighbor of the other Primitive

        This relation is symmetric, i.e. a.is_neighbor_of(b) <=> b.is_neighbor_of(a)
        """
        for neighbor in self.neighbors(predicate=predicate):
            if other is neighbor:
                return True
        else:
            return False

    def connect_neighbor(
        self,
        other: "Primitive",
        our_connector: Optional[ConnectorAddress | Connector] = None,
        their_connector: Optional[ConnectorAddress | Connector] = None,
        alignment_strategy: Optional[ConnectorAntialignmentStrategy] = None,
        n_iter_max_rule: Optional[GraphIterRule] = None,
    ) -> None:
        """
        Forge a new connection to another Primitive

        If explicit Connectors are provided for either or both Primitives,
        will use those as halves of the connection;
        Otherwise, will attempt to deduce a uiqnue choice using the linking algorithm
        """
        # to be used as keys identifying edge in graph
        prim_edge: tuple[Primitive, Primitive] = (self, other)
        mapped_connectors: dict[PrimitiveAddress, set[Connector]] = {
            self: set(self.connections.connectors_free)
            if our_connector is None
            else {self.fetch_connector(our_connector)},
            other: set(other.connections.connectors_free)
            if their_connector is None
            else {other.fetch_connector(their_connector)},
        }

        # TB: deducing, rather than assigning, to get access
        # to chosen Connectors for geometric alignment
        conn_map = deduce_connections_from_topology(
            topology=Graph([prim_edge]),
            mapped_connectors=mapped_connectors,
            n_iter_max_rule=n_iter_max_rule,
        )

        # extract pair (if found) and set as underlying neighbors
        our_connector_chosen, their_connector_chosen = conn_map[prim_edge]
        our_connector_chosen.neighbor = their_connector_chosen

        # TODO: align neighbor using chosen method
        # TODO: also align all neighbors of neighbor? (that connected to self elsewhere)
        if alignment_strategy is not None:
            alignment_strategy.antialign(
                align_connector=their_connector_chosen,
                to_connector=our_connector_chosen,
            )
            alignment_transform = alignment_strategy.antialignment_transformation(
                align_connector=their_connector_chosen,
                to_connector=our_connector_chosen,
            )  # Suppress on already-aligned chosen Connectors
            other.rigidly_transform(alignment_transform)

    # Hierarchy
    ## Enforcing universal hierarchy invariants
    # TB: the key invariants that must be enforced at all times are:
    # * Roots can never be the children of any other Primitive
    # * Simples can never be the parent of any other Primitive

    # N.B.: enforcing Simple-childfree and Root-parentfree is easy
    # to do directly within their respective class definitions;
    # the converses, parent-not-Simple and child-not-Root need to be enforced indirectly
    # here because of how anytree's pre/post-conditions are handled on assignment
    def _pre_attach(self, parent: "Primitive") -> None:
        if parent.is_simple:
            raise IrreducibilityError(
                "Simple Primitives cannot be made the parents of other Primitives"
            )

    def _pre_detach(self, parent: "Primitive") -> None:
        if parent.is_simple:
            raise IrreducibilityError(
                "Found hierarchy in undefined state, with "
                "Simple Primitive as parent of another Primitive"
            )

    def _pre_attach_children(self, children: Iterable["Primitive"]) -> None:
        # TODO: prevent roots from being assigned as children here
        ...

    def _pre_detach_children(self, children: Iterable["Primitive"]) -> None:
        # TODO: prevent roots from being unassigned as children here
        ...

    ## Inspection
    def search_hierarchy_by(
        self,
        predicate: PrimitivePredicate,
        halt_when: Optional[PrimitivePredicate] = None,
        to_depth: Optional[int] = None,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
    ) -> tuple["Primitive"]:
        """
        Return all Primitives below this one in the hierarchy (not just children,
        but anything below them as well!) which match the provided condition.

        Matching descendant Primitives are returned in traversal preorder from the root
        """
        return findall(
            self,
            filter_=predicate,
            stop=halt_when,
            maxlevel=to_depth,
            mincount=min_count,
            maxcount=max_count,
        )

    def hierarchy_summary(
        self,
        to_depth: Optional[int] = None,
        style: Union[str, ConcreteStyle, Type[ConcreteStyle]] = "round",
        render_attr: str = "label",
        # TB: may consider fallback to address (or start of it) instead of default label
    ) -> str:
        """
        A printable representation of this Primitive
        and all its descendants in the hierarchy
        """
        return RenderTree(
            self,
            style=tree_render_style(style),
            maxlevel=to_depth,
            # childiter=list
        ).by_attr(render_attr)

    def hierarchy_tree(self, *args) -> DiGraph:
        """Generate a directed Graph representing the hierarchy below this Primitive"""
        return anytree_to_networkx(self, *args)

    # Depiction
    # def __str__(self) -> str:
    #     # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
    #     # return self.canonical_form() # self.canonical_form_salted()
    #     raise NotImplementedError

    # def __repr__(self) -> str:
    #     raise NotImplementedError # TODO - will likely have to change for subtypes


class SupportsChildren(Primitive):
    """
    Type of Primitive which is allowed to have
    other Primitives "beneath" it in a hierarchy.

    I.e. interpreting a representation hierarchy as a rooted tree,
    these Primitives are nodes which allow OUTGOING directed edges
    """

    # Hierarchy
    ## Lookup
    children_by_address: WeakValueDictionary[PrimitiveAddress, "SupportsParents"]

    def child(self, prim_addr: PrimitiveAddress) -> "SupportsParents":
        """
        Lookup a child Primitive by its address and
        return the Primitive instance, if present
        """
        return self.children_by_address[prim_addr]  # raise KeyError if not present

    fetch_primitive = child

    ## Attachment
    def _pre_attach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Preconditions prior to attempting attachment of this Primitive to a parent"""
        super()._pre_attach_children(children)
        self._precondition_mutable_connectors()  # positions and neighbors may shift

        self._precondition_mutable_hierarchy()
        for child in children:
            child._precondition_mutable_hierarchy()

    def _post_attach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Post-actions to take once attachment is verified and parent is bound"""
        super()._post_attach_children(children)
        # TODO: remap connection info
        ...

    # TB: consider making just wrappers, with business logic
    # moved to _pre_attach/_post_attach conditions?
    def attach_child(
        self,
        child: "SupportsParents",
        label: Optional[PrimitiveLabel] = None,
    ) -> PrimitiveAddress:
        """
        Register a new child Primitive as existing below
        this one in the resolution hierarchy
        """
        child.parent = self
        # child.label = label
        self.children_by_address[child.address] = child

        for conn in child.connections.connectors:
            for superprimitive in self.path:
                superprimitive.connections.add_connector(
                    conn
                )  # requires Mutable manager, hence precondition on Connectors

        return child.address

    ## Detachment
    def _pre_detach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Preconditions prior to attempting detachment of this Primitive from parent"""
        self._precondition_mutable_hierarchy(
            msg="Hierarchy modification is forbidden on this Primitive; "
            "cannot detach extant outgoing node(s)"
        )

    def detach_child(self, prim_addr: PrimitiveAddress) -> Primitive:
        """
        Unregister an existing child Primitive, making it no
        be longer below this one in the resolution hierarchy
        """
        child = self.children_by_address.pop(prim_addr)
        child.parent = None

        for conn in child.connections.connectors:
            for superprimitive in self.path:
                superprimitive.connections.remove_connector(conn)
                # TB: what to do with Connectors' neighbors?

        return child

    ## Resolution shift operations
    def expand(self) -> None:
        """
        In the hierarchy, replace this Primitive in with its children,
        preserving connections and traces
        """
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    def flatten(self) -> None:
        """
        Recursively expand until all childless
        subprimitives are depth 1 below this one
        """
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    def contract(
        self,
        parts: Iterable[AbstractSet[PrimitiveAddress]],
        implicit_parts: bool = True,
    ) -> None:
        """
        Insert a new level of Primitive between this Composite and its children,
        with each part of the provided partition forming a new child Primitive

        Behavior of implicit parts (i.e. any not explicitly mentioned in "parts")
        can be specified via the "implicit_parts" argument
        """  # DEV: eventually, make enum for implicit_parts behavior
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    def truncate(self) -> None:
        """
        Replace this Composite with an analogous Simple,
        disconnecting all its children from the rest of the hierarchy tree
        """
        self._precondition_mutable_hierarchy()
        raise NotImplementedError

    # Geometry
    ## Overriding RigidlyTransformable contracts - apply recursively to children as well
    def _copy_untransformed(self) -> "Primitive":
        raise NotImplementedError

    def _rigidly_transform(self, transformation: RigidTransform) -> None:
        raise NotImplementedError

    # Topology
    def set_connectivity_from_topology(
        self,
        topology: Graph,
        predicate: PrimitivePredicate,
        n_iter_max_rule: Optional[Callable[[int], int]] = None,
    ) -> None:
        """
        Form connections from a labelled graph,
        paying respect to selectivity of Connectors
        """
        assign_connections_from_topology(
            topology,
            mapped_connectors={
                # TODO: figure out how to map from unique addresses to graph node
                subprim.addr: subprim.connections.connectors
                for subprim in select_primitives(
                    self.descendants,
                    predicate=predicate,
                )
            },
            n_iter_max_rule=n_iter_max_rule,
        )

    def cross_section(self, predicate: PrimitivePredicate) -> Graph:
        """
        Generate a graph of a "slice" of a subset
        of sub-Primitives specified by a predicate
        """
        multigraph_conversion_made: bool = False

        cross_section = Graph()
        cross_section.add_nodes_from(
            select_primitives(
                self.descendants,
                predicate=predicate,
            )
        )
        visited: dict[Primitive, bool] = dict()
        for prim_node in cross_section.nodes:
            seen_neighbors: set[Primitive] = set()
            for neighbor in prim_node.neighbors(predicate):
                if visited.get(neighbor, False):
                    continue

                # upconvert to multigraph the first time a duplicate edge is encoutered
                if (not multigraph_conversion_made) and (neighbor in seen_neighbors):
                    cross_section = MultiGraph(cross_section)
                    multigraph_conversion_made = True

                cross_section.add_edge(prim_node, neighbor)
                seen_neighbors.add(neighbor)
            visited[prim_node] = True  # avoids double-counting single edges

        return cross_section


class SupportsParents(Primitive):
    """
    Type of Primitive which is allowed to have
    other Primitives "above" it in a hierarchy

    I.e. interpreting a representation hierarchy as a rooted tree,
    these Primitives are nodes which allow INCOMING directed edges
    """

    # Hierarchy

    # TB: you might be thinking it would be more natural to have the
    # checks on parent Primitives in SupportParent instead
    # the reason for having them here instead is that setting children
    # always calls `child.parent = new_parent_value` under the hood
    def _pre_attach(self, parent: SupportsChildren) -> None:
        super()._pre_attach(parent)
        self._precondition_mutable_hierarchy()
        parent._precondition_mutable_hierarchy()

    def _post_attach(self, parent: SupportsChildren) -> None: ...

    def _pre_detach(self, parent: SupportsChildren) -> None:
        super()._pre_detach(parent)
        self._precondition_mutable_hierarchy()
        parent._precondition_mutable_hierarchy()

    def _post_detach(self, parent: SupportsChildren) -> None: ...


# Concrete primitive types
## Tree root
class RootPrimitive(SupportsChildren):
    """
    Base of a hierarchy tree - no Primitives can exist above (i.e. own) this one
    Used to store system-wide metadata, as well as provide hand-off point for interfaces
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "ROOT"

    def __init__(
        self,
        box_vectors: Optional[Array3x3] = None,
        shape: Optional[BoundedTransformableShape] = None,
        metadata: Optional[dict[Hashable, Any]] = None,
    ) -> None:
        self.connections = ConnectorManagerMutable()
        self._shape = shape
        self.metadata = metadata or dict()

        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

        # implements SupportsChildren contract
        self.children_by_address = WeakValueDictionary()

        # system-wide info specific to Root instances
        if box_vectors is None:
            # TODO: associate units (once a standard has been decided upon)
            box_vectors = np.eye(3, dtype=float)
        self.box_vectors = box_vectors

    # DEV: deliberately excluded public setter for is_frozen;
    # this should never be tampered with externally

    # Managing hierarchy
    ## Explicitly banning parents
    def _pre_attach(self, parent: SupportsChildren) -> None:
        super()._pre_attach(parent)
        raise ArborescenceError(
            "Cannot make Root of hierarchy the child of another Primitive"
        )

    def _pre_detach(self, parent: SupportsChildren) -> None:
        super()._pre_detach(parent)
        raise ArborescenceError(
            "Invalid state: Root is somehow the child of another Primitive"
        )


## Composites
class CompositePrimitive(SupportsChildren, SupportsParents):
    """
    Primitive representing intermediate levels of organization in a chemical system;
    In a representation hierarchy, always lives between Roots and Simples
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "COMPOSITE"

    def __init__(
        self,
        children: Optional[Iterable[SupportsParents]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._shape = shape
        self.metadata = metadata or dict()
        self.connections = ConnectorManagerMutable()

        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

        # Binding initial subprimitives
        self.children_by_address = (
            WeakValueDictionary()
        )  # implements SupportsChildren contract
        if children is None:
            children = tuple()

        for subprimitive in children:
            self.attach_child(subprimitive, label=subprimitive.label)

    # Hierarchy
    ...

    ## Topology
    ...


## Simples
class SimplePrimitive(SupportsParents):
    """
    A Primitive with no internal structure
    i.e. no children, topology, or internal connections)

    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "SIMPLE"

    def __init__(
        self,
        connections: Optional[ConnectorManager | Iterable[Connector]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        metadata: Optional[dict[Hashable, Any]] = None,
    ) -> None:
        # TB: have to be careful in this typecheck if ConnectorManager is also Iterable
        if isinstance(connections, Iterable):
            connections = ConnectorManagerMutable(*connections)
        elif connections is None:
            connections = ConnectorManagerMutable()

        self.connections = connections
        for connector in connections.connectors:
            connector.holder = self

        self._shape = shape
        self.metadata = metadata or dict()

        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

    @property
    def is_simple(self) -> bool:
        """
        Asserts SimplePrimitives are indivisible from
        the prespective of a hierarchy of Primitives

        Bans SimplePrimitives from being the parent of any other Primitive,
        or conversely of having any child Primitives
        """
        # override from Primitive base; only class which should do so
        return True

    # Exposing Connectors
    def inject_connector_into_hierarchy(
        self,
        connector: Connector,
    ) -> ConnectorAddress:
        """
        Introduce a new Connector into circulation throughout the hierarchy above

        All ancestors of this Simple will also manage this Connector instance
        """
        for anc in self.ancestors:
            anc.connections.add_connector(connector)

    def add_connector(
        self,
        connector: Connector,
        label: Optional[ConnectorLabel] = None,
    ) -> None:
        """Add a new Connector to those managed by this Simple"""
        self._precondition_mutable_connectors()
        self.connections.add_connector(
            connector,
            # TB: label is irrelevant w/ addresses; keeping
            # only in case handles prove useful to add later
            label=(label or Connector.DEFAULT_LABEL),
        )
        connector.holder = self

        self.inject_connector_into_hierarchy(connector)

    def withdraw_connector_from_hierarchy(
        self,
        connector_address: ConnectorAddress | Connector,
        preserve_neighbor: bool = False,
    ) -> None:
        """
        Remove a Connector from all levels of the hierarchy above this Simple

        Connector will still be managed within this Simple,
        but with its former neighbor (if any) severed

        Returns the withdrawn Connector instance
        """
        connector_address = connector_address_flexible(connector_address)
        for ancestor in self.ancestors:
            # TB: these all point to the same Connector instance, so collecting
            # is technically redundant for all but the last iter of the loop
            connector = ancestor.connections.remove_connector(connector_address)

        if not preserve_neighbor:
            del connector.neighbor

    def remove_connector(
        self,
        connector_address: ConnectorAddress | Connector,
    ) -> Connector:
        """
        AND from being managed by this Simple itself
        AND remove a Connector from all levels of the hierarchy above this Simple
        I.e. the passed Connector will be completely removed from the managing hierarchy

        Returns the removed Connector
        """
        self._precondition_mutable_connectors()

        connector_address = connector_address_flexible(connector_address)
        connector = self.connections.remove_connector(connector_address)
        del connector.holder  # will be self, since this Simple is at end of Path

        self.withdraw_connector_from_hierarchy(
            connector_address,
            # never want to leave dangling neighbors if completely removed
            preserve_neighbor=False,
        )
        return connector

    # Hierarchy
    def _post_attach(self, parent: SupportsChildren) -> None:
        super()._post_attach(parent)
        for connector in self.connections.connectors:
            self.inject_connector_into_hierarchy(connector)

    def _post_detach(self, parent: SupportsChildren) -> None:
        super()._post_attach(parent)
        for connector in self.connections.connectors:
            self.withdraw_connector_from_hierarchy(connector, preserve_neighbor=False)

    ## Explicit ban on attachment of children (already simple)
    def _pre_attach_children(self, children: Iterable[Primitive]) -> None:
        super()._pre_attach_children(children)
        raise IrreducibilityError(
            f"Simple Primitives cannot be assigned children {children}"
        )

    def _pre_detach_children(self, children: Iterable[Primitive]) -> None:
        super()._pre_detach_children(children)
        raise IrreducibilityError(
            "Found hierarchy in undefined state, with "
            f"Simple Primitive as parent of {children}"
        )

    ## TODO: register all Connectors held by self to parent and all ancestors once set


class AtomicPrimitive(SimplePrimitive):
    """
    A Primitive representing a single atom from the periodic table
    Contains element, formal charge, and nuclear mass information about the atom
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "ATOM"

    def __init__(
        self,
        element: ElementLike,
        connections: Optional[ConnectorManager] = None,
        shape: Optional[BoundedTransformableShape] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if not isatom(element):
            raise TypeError(f"Invalid element type {type(element)}")
        self._element = element

        super().__init__(
            connections=connections,
            shape=shape,
            metadata=metadata,
        )

    @property  # DEV: no setter implemented; element is immutable after instantiation
    def element(self) -> ElementLike:
        """The chemical element, ion, or isotope associated with this AtomicPrimitive"""
        return self._element

    def check_valence(self) -> None:
        """
        Check that element assigned to atomic Primitives and
        bond orders of Connectors are chemically-compatible
        """
        valence: float = self.connections.valence
        if not valence_allowed(
            self.element.number,
            self.element.charge,
            valence,
        ):
            raise ValueError(
                f"Atomic {self!r} with total valence {valence} "
                "incompatible with assigned element {self.element!r}"
            )

    # def canonical_form(self) -> str:
    #     return f'{self.element.symbol}{canonical_form_primitive(self)}'


# Hashable canonical forms for core components
def canonical_form_shape(primitive: Primitive) -> str:
    """A canonical string representing this Primitive's shape"""
    # TODO: move this into .shape; should be responsibility of Shape subclasses
    return type(primitive.shape).__name__


def canonical_form_primitive(
    primitive: Primitive,
) -> (
    str
):  # NOTE: deliberately NOT a property to indicated computing this might be expensive
    """
    A canonical representation of a Primitive's core parts;
    induces a natural equivalence relation on Primitives

    I.e. two Primitives having the same canonical form are
    to be considered interchangable within a polymer system
    """
    return (
        f"(connectors={canonical_form_connectors(primitive.connections.connectors)})"
        f"[shape={canonical_form_shape(primitive)}]"
    )
    # f'<graph_hash={self.canonical_form_topology()}>'
