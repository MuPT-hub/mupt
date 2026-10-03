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
    TypeVar,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

H = TypeVar("H", bound=Hashable)
type PrimitiveLabel = Hashable
type PrimitiveAddress = Hashable
type PrimitiveHandle = tuple[PrimitiveLabel, int]  # (label, uniquification index)

from copy import deepcopy
from weakref import WeakValueDictionary

from anytree.node import NodeMixin
from anytree.render import RenderTree
from anytree.search import findall
from anytree.iterators import LevelOrderIter

from networkx.classes import Graph, DiGraph, MultiGraph
from networkx import get_node_attributes, relabel_nodes

import numpy as np
from scipy.spatial.transform import RigidTransform

from .connection.types import (
    ConnectorAddress,
    ConnectorLabel,
    ConnectorLabelLike,
)
from .connection.connectors import (
    Connector,
    canonical_form_connectors,
)
from .connection.management import (
    ConnectorManager,
    ConnectorManagerFrozen,
    ConnectorManagerMutable,
    connector_address_flexible,
)
from .connection.alignment import (
    ConnectorAntialignmentStrategy,
    ConnectorAntialignmentBallistic,
)
from .connection.linking import (
    deduce_connections_from_topology,
    assign_connections_from_topology,
    GraphIterRule,
)

from ..trees.digraph import anytree_to_networkx
from ..trees.subselect import primoprogenitors, NodePredicate
from ..trees.render import tree_render_style, ConcreteStyle
from ..graphs.visualisation import draw_networkx_with_arcs

from ..mutils.referencing import Addressed
from ..mutils.containers import Labelled
from ..geometry.arraytypes import Array3x3
from ..geometry.shapes import Shaped, BoundedTransformableShape
from ..geometry.transforms.rigid import RigidlyTransformable
from ..chemistry.core import BOND_ORDER_ATTR, ElementLike, isatom, valence_allowed


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
def indiscriminate_selector(prim: "Primitive") -> bool:
    """
    Selector which always greenlights the passed Primitive no matter what
    Useful for avoiding lamba overhead
    """
    return True


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
    connections: ConnectorManager
    metadata: dict[Hashable, Any]
    _shape: Optional[BoundedTransformableShape]  # TODO: add protected access
    _label: PrimitiveLabel

    _frozen_connections: bool
    _frozen_hierarchy: bool

    ## Derived properties
    @property
    def label(self) -> PrimitiveLabel:
        """
        A distinguishing label which can be assigned
        by the user for identification purposes
        """
        if self._label is None:
            # Attempt fallbacks to fetch label
            if "label" in self.metadata:
                self._label = self.metadata["label"]
            else:
                self._label = self.DEFAULT_LABEL

        return self._label

    @label.setter
    def label(self, new_label: Optional[PrimitiveLabel]) -> None:
        """Assign a new label to this Primitive"""
        if new_label is None:
            new_label = self.DEFAULT_LABEL
        self._label = new_label

    @property
    def is_simple(self) -> bool:
        """
        Whether Primitives are to be considered indivisible
        from the perspective of the hierarchy
        """
        # DEVNOTE: this is a mechanism to prevent Simples from being the parents of any
        # other Primitive without passing type info backward up the inheritance tree
        return False

    ## Wrapped properties
    @property
    def connectors(self) -> Collection[Connector]:
        """Convenience wrapper for accessing ALL connectors managed by this Primitive"""
        # TODO: also provide convenient access to connectors_free and connectors_bound
        return self.connections.connectors

    @property
    def functionality(self) -> int:
        """Number of free Connections this Primitive has access to"""
        return self.connections.functionality

    # Mutability flags
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
    @property
    def shape(self) -> Optional[BoundedTransformableShape]:
        """The external shape of this Primitive"""
        return self._shape

    def _copy_instance(self) -> Self:
        """
        Make copy of the current Primitive, WITHOUT any
        hierarchical or Connection information included
        """
        raise NotImplementedError  # TB TODO: should be abstractmethod, eventually

    def _copy_hierarchy(self) -> tuple[Self, dict["Primitive", "Primitive"]]:
        """
        Copy the current Primitive and the hierarchy of all its Primitive ancestors

        Return the root of the copied hierarchy (a Primitive analogous to this one)
        and a map from the existing Primitives addresses to the newly-created ones
        """
        # TB: convert parts to address-based reference, if possible?
        orig_prim_to_new_prim: dict[Primitive, Primitive] = dict()

        for subprim in LevelOrderIter(self):
            clone_no_hierarchy_subprim = subprim._copy_instance()
            orig_prim_to_new_prim[subprim] = clone_no_hierarchy_subprim

            new_parent: Optional[SupportsChildren] = orig_prim_to_new_prim.get(
                subprim.parent, None
            )
            clone_no_hierarchy_subprim.parent = new_parent

        return orig_prim_to_new_prim[self], orig_prim_to_new_prim

    def _copy_untransformed(self) -> Self:
        # TB: intentionally left blank; while generic _rigidly_transform
        # is possible to implement in here in the base, the specifics of
        # creating new instances must be up to the concrete subtypes
        """
        Create a copy which has a complete hierarchy and connectivity below it
        Does not apply any rigid transformation to itself or sub-components

        Connections outside the "cone" below this Primtive will be severed in the copy
        """
        clone_with_hierarchy, orig_prim_to_copy = self._copy_hierarchy()
        orig_conn_to_copy: dict[Connector, Connector] = dict()

        for connector in self.connections.connectors:
            connector_copy = connector.copy()
            del connector_copy.neighbor  # just to be safe

            # DEV: not using dict.get(), since we WANT a KeyError if holders are unset
            new_holder = orig_prim_to_copy[connector.holder]
            # add_connector should propagate Connector copies through all Primitives
            # in the copied hierarchy, including it among *this* clone's connectors
            new_holder.add_connector(connector_copy)

            orig_conn_to_copy[connector] = connector_copy

        # not using dict.items() to avoid calamity from dict modification during iter
        # needed to prevent double-counting bonds via their two constituent Connectors
        connectors_to_visit: set[Connector] = set(orig_conn_to_copy)
        while connectors_to_visit:
            # regardless of neighbor status, remove *this* Connector from search pool
            orig_connector = connectors_to_visit.pop()
            if not orig_connector.has_neighbor:
                continue

            # Connector is "internal" <=> its neighbor is also managed by this Primitive
            if (orig_neighbor := orig_connector.neighbor) in orig_conn_to_copy:
                copy_connector = orig_conn_to_copy[orig_connector]

                # to avoid double-counting bonds, don't visit original neighbor later
                copy_connector.neighbor = orig_conn_to_copy.pop(orig_neighbor)
                connectors_to_visit.remove(orig_neighbor)

        return clone_with_hierarchy

    def _rigidly_transform_shape(self, transformation: RigidTransform) -> None:
        """Apply rigid transformation to just the shape of this Primitive"""
        if isinstance(self.shape, RigidlyTransformable):  # TB: just check if not None?
            self.shape.rigidly_transform(transformation)

    def _rigidly_transform_connectors(self, transformation: RigidTransform) -> None:
        """Apply rigid transformation to just the Conenctors managed by Primitive"""
        # DEV: this should NOT be a configurable arg; always want ballistic here
        antialign_strategy = ConnectorAntialignmentBallistic()
        for connector in self.connections.connectors:
            connector.rigidly_transform(transformation)
            if connector.has_neighbor:
                antialign_strategy.antialign(
                    align_connector=connector,
                    to_connector=connector.neighbor,  # keep neighbor fixed
                    match_bond_length=True,
                    dihedral_angle_rad=None,  # may configure in future
                )

    def _rigidly_transform(self, transformation: RigidTransform) -> None:
        """Apply a rigid transformation to all parts of a Primitive which support it"""
        self._rigidly_transform_connectors(transformation)
        self._rigidly_transform_shape(transformation)
        for subprimitive in self.descendants:  # descendants avoids recursive calls
            # N.B.: not transforming sub-primitives' Connectors since they are, in
            # aggregrate THE SAME Connectors managed here (don't double-transform)
            subprimitive._rigidly_transform_shape(transformation)

    # Topology
    ## Connection read/write access
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

    def fetch_connector(self, conn: ConnectorAddress | Connector) -> Connector:
        """Fetch a connector managed by this Priomitive, if it exists"""
        return self.connections.connector(connector_address_flexible(conn))

    # N.B.: deliberately private; only Simples should have explicit access
    def _add_connector(
        self,
        connector: Connector,
        label: Optional[ConnectorLabelLike] = None,
    ) -> ConnectorAddress:
        """Add a new Connector to those managed locally"""
        self._precondition_mutable_connectors()
        # TB: label is irrelevant w/ addresses; keeping
        # only in case handles prove useful to add later
        self.connections.add_connector(connector, label=label)
        # N.B.: connector.holder deliberately unset here

        return connector.address

    def _remove_connector(
        self,
        connector_address: Connector | ConnectorAddress,
    ) -> Connector:
        """Remove an existing Connector from those managed locally"""
        self._precondition_mutable_connectors()
        # N.B.: connector neighbor deliberately untouched here
        return self.connections.remove_connector(connector_address)

    ## Adjacency
    def neighbors_with_connectors(
        self,
        predicate: Optional[NodePredicate["Primitive"]] = None,
    ) -> Generator[tuple[Connector, Connector, "Primitive"], None, None]:
        """
        Generates the two Connectors which constitute a connection
        to a neighboring Primitive, as well as that Primitive itself
        Yields as (our_connector, their_connector, them) tuples

        Can sub-select among resolutions of neighbor
        branches using a predicate, if provided
        """
        if predicate is None:
            # N.B.: opting for this mechanism for default predicate, rather than
            # setting indiscriminate_selector as arg default, to avoid external
            # consumer needing to know about default impl (i.e. can just pass None)
            predicate = indiscriminate_selector

        for our_connector in self.connections.connectors_bound:
            # TB TODO: finesse typehints to suppress (perceived) unset NoneType values
            their_connector: Connector = our_connector.neighbor
            neighbor_leaf: Primitive = their_connector.holder
            neighbor_branch: tuple[Primitive] = neighbor_leaf.path

            if self in neighbor_branch:
                # avoid "internal" neighbors (of whom *this* Primitive is also a parent)
                continue

            # any superprimitives which share connectors with the holder are
            # also considered neighbors; this is what enables multiscaling
            for neighbor in neighbor_branch:
                if predicate(neighbor):
                    # TB: use primoprogenitors() here? I.e. do we want to allow
                    # multiple neighbors from the same parallel branch here?
                    yield our_connector, their_connector, neighbor

    def neighbors(
        self,
        predicate: Optional[NodePredicate["Primitive"]] = None,
    ) -> Generator["Primitive", None, None]:
        """
        Primitives whose share a Connection with this one

        Can sub-select among resolutions of neighbor
        branches using a predicate, if provided
        """
        for _, _, neighbor in self.neighbors_with_connectors(predicate=predicate):
            yield neighbor

    def is_neighbors_with(
        self,
        other: "Primitive",
        predicate: Optional[NodePredicate["Primitive"]] = None,
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

        If explicit Connectors are provided for either or both
        Primitives,will use those as halves of the connection;
        Otherwise, will attempt to deduce a unique choice using the linking algorithm
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

    def cross_section(self, predicate: NodePredicate["Primitive"]) -> Graph:
        """
        Generate a graph of a "slice" of a subset
        of sub-Primitives specified by a predicate
        """
        multigraph_conversion_made: bool = False

        cross_section = Graph()
        cross_section.add_nodes_from(primoprogenitors(self, predicate=predicate))

        visited: dict[Primitive, bool] = dict()
        for prim_node in cross_section.nodes:
            seen_neighbors: set[Primitive] = set()
            for (
                our_connector,
                their_connector,
                neighbor,
            ) in prim_node.neighbors_with_connectors(predicate):
                if visited.get(neighbor, False):
                    continue

                # upconvert to multigraph the first time a duplicate edge is encoutered
                if (not multigraph_conversion_made) and (neighbor in seen_neighbors):
                    cross_section = MultiGraph(cross_section)
                    multigraph_conversion_made = True

                # should already match if Connectors were allowed to be neighbors,
                # but it never hurts to double-check
                assert our_connector.bond_order == their_connector.bond_order

                cross_section.add_edge(
                    prim_node,
                    neighbor,
                    **{BOND_ORDER_ATTR: our_connector.bond_order},
                )
                seen_neighbors.add(neighbor)

            # avoids double-counting single edges
            visited[prim_node] = True

        return cross_section

    def visualize_cross_section(
        self,
        cross_section: Union[Graph, NodePredicate["Primitive"]],
        base_arc_radius: float = 0.1,
        # TODO: provide comprehensive typehint for all things mpl can accept as colors
        coloring_rule: Optional[Callable[["Primitive"], str]] = None,
        **kwargs,
    ) -> "Axes":
        """
        Draw a networkx graph representation of the selected cross-section

        Can accept a pre-calculated cross-section or, if none is provided
        will calculate the cross section on-the-spot before plotting
        """
        if not isinstance(cross_section, Graph):
            LOGGER.info(
                "Extracting cross-section from predicate, "
                "as no pre-computed cross-section was provided"
            )
            cross_section = self.cross_section(cross_section)

        if coloring_rule is not None:
            kwargs["node_color"] = [coloring_rule(prim) for prim in cross_section]

        return draw_networkx_with_arcs(
            cross_section, base_arc_radius=base_arc_radius, **kwargs
        )

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
        predicate: NodePredicate["Primitive"],
        halt_when: Optional[NodePredicate["Primitive"]] = None,
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

    def hierarchy_tree(self, *args, **kwargs) -> DiGraph:
        """Generate a directed Graph representing the hierarchy below this Primitive"""
        return anytree_to_networkx(self, *args, **kwargs)

    # Depiction
    def __str__(self) -> str:
        """
        Output of calling str(...) on this Primitive

        Also the default representation of this Primitive
        when it is used as a node in any NetworkX graph
        """
        return f"{self.label!s}[{str(self.address)[:7]}]"

    # def __repr__(self) -> str:
    #     # DEV: will likely have to change for subtypes
    #     raise NotImplementedError


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

    def _init_children(
        self,
        children: Optional[Iterable["SupportsParents"]] = None,
    ) -> None:
        """
        Perform one-time initialization of children-related attributes and, if
        provided, bind children to self from collection of parent-capable Primitives
        """
        self.children_by_address = WeakValueDictionary()
        if children is None:
            children = tuple()

        for subprimitive in children:
            self.attach_child(subprimitive, label=subprimitive.label)

    def child(self, prim_addr: PrimitiveAddress) -> "SupportsParents":
        """
        Lookup a child Primitive by its address and
        return the Primitive instance, if present
        """
        return self.children_by_address[prim_addr]  # raise KeyError if not present

    fetch_primitive = child

    ## Attachment
    def _pre_attach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Preconditions prior to attempting to attach of this Primitive to a parent"""
        super()._pre_attach_children(children)
        self._precondition_mutable_connectors()  # positions and neighbors may shift
        self._precondition_mutable_hierarchy()

        for child in children:
            child._precondition_mutable_hierarchy()

    def _post_attach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Post-actions to take once children are attached and parent is bound"""
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

        return child.address

    ## Detachment
    def _pre_detach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Preconditions prior to attempting detachment of this Primitive from parent"""
        self._precondition_mutable_hierarchy(
            msg="Hierarchy modification is forbidden on this Primitive; "
            "cannot detach extant outgoing node(s)"
        )

    def _post_detach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Post-actions to take once children are detached and parent is unbound"""
        super()._post_detach_children(children)

    def detach_child(self, prim_addr: PrimitiveAddress) -> Primitive:
        """
        Unregister an existing child Primitive, making it no
        be longer below this one in the resolution hierarchy
        """
        child = self.children_by_address.pop(prim_addr)
        child.parent = None

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

    # Topology
    def set_connectivity_from_topology(
        self,
        topology: Graph,  # Graph[H]
        predicate: NodePredicate["Primitive"],
        prim_node_labeller: Callable[[Primitive], H] = lambda prim: prim,
        n_iter_max_rule: Optional[GraphIterRule] = None,
        source_node: Optional[H] = None,
    ) -> None:
        """
        Form connections from a labelled graph, respecting selectivity of Connectors

        Parameters
        ----------
        topology: Graph
            The graph to use to assign neighbor connectviity
        predicate: NodePredicate["Primitive"]
            The condition by which to select sub-Primitives
        prim_node_labeller : Callable[[Primitive], Hashable] /
                default lambda prim : prim,
            A function which maps the selected sub-Primitives to hashable labels
            The labels mapped to should match nodes of the passed graph
            By default, just returns the Primitive instance itself
        n_iter_max_rule: Optional[Callable[[int], int]] = None
            A rule for assigning the max number of iterations the linker routine
            should run before giving up, as a function of the passed graph
        """
        assign_connections_from_topology(
            topology,
            mapped_connectors={
                prim_node_labeller(subprim): subprim.connections.connectors_free
                for subprim in primoprogenitors(self, predicate=predicate)
            },
            n_iter_max_rule=n_iter_max_rule,
            source_node=source_node,
        )

    def populate_from_topology_and_lexicon(
        self,
        topology: Graph,
        lexicon: dict[PrimitiveLabel, "SupportsParents"],
        label_attr: str = "label",
        source_node_label: Optional[PrimitiveLabel] = None,
    ) -> None:
        """
        Populate the internal structure of this child-supporting Primitive
        by assigning its children and connectivity from a labelled graph
        and a 'lexicon' mapping from labels to Primitive templates
        """
        label_to_prim_map: dict[PrimitiveLabel, SupportsParents] = {
            node: lexicon[label_value].copy()
            for node, label_value in get_node_attributes(
                topology,
                label_attr,
            ).items()
        }

        prim_topology = relabel_nodes(topology, label_to_prim_map, copy=True)
        for subprim in prim_topology.nodes:
            self.attach_child(subprim)

        self.set_connectivity_from_topology(
            prim_topology,
            predicate=lambda prim: prim in prim_topology,
            prim_node_labeller=lambda x: x,
            source_node=label_to_prim_map.get(source_node_label, None),
        )


class SupportsParents(Primitive):
    """
    Type of Primitive which is allowed to have
    other Primitives "above" it in a hierarchy

    I.e. interpreting a representation hierarchy as a rooted tree,
    these Primitives are nodes which allow INCOMING directed edges
    """

    # Topology
    def inject_connector_into_hierarchy(
        self,
        connector: Connector,
        label: Optional[ConnectorLabelLike] = None,
    ) -> ConnectorAddress:
        """
        Introduce a new Connector into circulation throughout the hierarchy above
        All ancestors of this Primitive will also manage this Connector instance

        Returns the address of the injected Connector
        """
        for anc in self.ancestors:
            anc._add_connector(connector, label=label)
        return connector.address

    def withdraw_connector_from_hierarchy(
        self,
        connector_address: ConnectorAddress | Connector,
        preserve_neighbor: bool = False,
    ) -> Connector:
        """
        Remove a Connector from circulation in levels of the hierarchy above
        Connector will still be managed within THIS Primitive,

        Neighbor of Connector will be severed by default to prevent corruption of
        hierarchy; can manually override if desired by passing "preserve_neighbor=True"

        Returns the withdrawn Connector
        """
        connector_address = connector_address_flexible(connector_address)
        for ancestor in self.ancestors:
            # TB: these all point to the same Connector instance, so assigning to
            # var is technically redundant for all but the last iter of the loop
            connector = ancestor._remove_connector(connector_address)

        if not preserve_neighbor:
            del connector.neighbor
        return connector

    # Hierarchy
    # TB: you might be thinking it would be more natural to have checks on parent
    # Primitives in SupportParent instead; the reason for having them here instead is
    # setting children always calls `child.parent = new_parent_value` under the hood
    def _pre_attach(self, parent: SupportsChildren) -> None:
        """Ensure both parent and child are fully mutable"""
        super()._pre_attach(parent)
        self._precondition_mutable_hierarchy()
        parent._precondition_mutable_hierarchy()

    def _post_attach(self, parent: SupportsChildren) -> None:
        """Once parent is set, inject own Connectors into hierarchy"""
        super()._post_attach(parent)
        for connector in self.connectors:
            self.inject_connector_into_hierarchy(connector)

    def _pre_detach(self, parent: SupportsChildren) -> None:
        """Ensure both parent and child are fully mutable"""
        super()._pre_detach(parent)
        self._precondition_mutable_hierarchy()
        parent._precondition_mutable_hierarchy()

    def _post_detach(self, parent: SupportsChildren) -> None:
        super()._post_detach(parent)
        for connector in self.connectors:
            self.withdraw_connector_from_hierarchy(connector)


# Concrete primitive types
## Tree root
class RootPrimitive(SupportsChildren):
    """
    Base of a hierarchy tree - no Primitives can exist above (i.e. own) this one
    Used to store system-wide metadata, as well as provide hand-off point for interfaces
    """

    box_vectors: Array3x3

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "ROOT"

    def __init__(
        self,
        box_vectors: Optional[Array3x3] = None,
        children: Optional[Iterable[SupportsParents]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        metadata: Optional[dict[Hashable, Any]] = None,
        label: Optional[PrimitiveLabel] = None,
    ) -> None:
        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

        self.connections = ConnectorManagerMutable()
        self._shape = shape
        self.metadata = metadata or dict()
        self.label = label

        # N.B.: can't call before _frozen_hierarchy is set
        self._init_children(children)

        # implements SupportsChildren contract
        self.children_by_address = WeakValueDictionary()

        # system-wide info specific to Root instances
        if box_vectors is None:
            # TODO: associate units (once a standard has been decided upon)
            box_vectors = np.eye(3, dtype=float)
        self.box_vectors = box_vectors

    # Copying
    def _copy_instance(self) -> Self:
        """
        Make a copy of this RootPrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            box_vectors=self.box_vectors.copy(),
            children=[],
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone

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
        label: Optional[PrimitiveLabel] = None,
    ) -> None:
        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

        self._shape = shape
        self.metadata = metadata or dict()
        self.connections = ConnectorManagerMutable()
        self.label = label

        # N.B.: can't call before _frozen_hierarchy is set
        self._init_children(children)

    # Copying
    def _copy_instance(self) -> Self:
        """
        Make a copy of this CompositePrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            children=[],
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone


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
        label: Optional[PrimitiveLabel] = None,
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
        self.label = label

        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

    # Copying
    def _copy_instance(self) -> Self:
        """
        Make a copy of this SimplePrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            connections=[],  # will be plumbed up in _copy_untransformed()
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone

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
    def add_connector(
        self,
        connector: Connector,
        label: Optional[ConnectorLabel] = None,
    ) -> ConnectorAddress:
        """
        Add a new Connector to those managed by this Simple

        Automatically assigns this Simple as its holder and propagates the
        Connector up through the hierarchy if, this Simple has a parent
        """
        connector_address = self._add_connector(connector, label=label)
        self.inject_connector_into_hierarchy(connector)
        connector.holder = self  # do last, in case above fails

        return connector_address

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
        connector = self._remove_connector(connector_address)
        self.withdraw_connector_from_hierarchy(
            connector_address, preserve_neighbor=False
        )
        del connector.holder  # will be self, since this Simple is at end of Path

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

    ## Explicitly bans attaching children to Simples
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
        connections: Optional[ConnectorManager | Iterable[Connector]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        metadata: Optional[dict] = None,
        label: Optional[PrimitiveLabel] = None,
    ) -> None:
        if not isatom(element):
            raise TypeError(f"Invalid element type {type(element)}")
        self._element = element

        super().__init__(
            connections=connections,
            shape=shape,
            metadata=metadata,
            label=label,
        )

    # Copying
    def _copy_instance(self) -> Self:
        """
        Make a copy of this AtomicPrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            element=self.element,  # TB: double-check this is actually a singleton
            connections=[],  # will be plumbed up in _copy_untransformed()
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone

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
