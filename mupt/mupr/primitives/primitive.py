"""
Base classes for Primitives, the fundamental datatype
within MuPT's multiscale molecular representation
"""

from typing import (
    Any,
    ClassVar,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Optional,
    Self,
    Union,
)

type PrimitiveLabel = Hashable
type PrimitiveAddress = Hashable
type Metadata = dict[Hashable, Any]

from weakref import WeakValueDictionary

from anytree.node import NodeMixin
from anytree.iterators import LevelOrderIter
from anytree.search import findall
from anytree.render import RenderTree
from anytree.util import commonancestors

from networkx.classes import DiGraph

from scipy.spatial.transform import RigidTransform

from .exceptions import IrreducibilityError
from ..connection.types import ConnectorAddress, ConnectorLabelLike
from ..connection.connectors import Connector
from ..connection.management import (
    connector_address_flexible,
    ConnectorManager,
    ConnectorManagerFrozen,
    ConnectorManagerMutable,
)
from ..connection.alignment import (
    ConnectorAntialignmentStrategy,
    ConnectorAntialignmentBallistic,
)

from ...mutils.referencing import Addressed
from ...mutils.containers import Labelled
from ...mutils.decorators.classmod import precondition_factory

from ...geometry.shapes import Shaped, BoundedTransformableShape
from ...geometry.transforms.rigid import RigidlyTransformable

from ...trees.digraph import anytree_to_networkx
from ...trees.subselect import NodePredicate, primoprogenitors
from ...trees.render import tree_render_style, ConcreteStyle


# Primitive selection
def indiscriminate_selector(prim: "Primitive") -> bool:
    """
    Selector which always greenlights the passed Primitive no matter what
    Useful for avoiding lamba overhead
    """
    # TB: move this to .properties? (be careful to avoid circular references)
    return True


DEFAULT_PREDICATE: NodePredicate["Primitive"] = indiscriminate_selector


class Primitive(
    Addressed,  # N.B.: each subtype will have its own class-wide registry
    Labelled,
    Shaped,
    RigidlyTransformable,
    NodeMixin,
):
    """A fundamental, scale-agnostic building block of a molecular system"""

    # Class attributes
    ## Global
    DEFAULT_LABEL: ClassVar[PrimitiveLabel]
    ADDRESS_PREVIEW_LEN: ClassVar[int] = 7

    ## Subtype-specific
    # TB: considered dropping the Law of the Excluded Middle here, i.e. via an
    # Optional[bool] type annotation, since strictly-speaking these properties are
    # UNDEFINED on the base, being neither true nor false
    # Opting not to now for type simplicity, since None is Falsy anyways
    IS_SIMPLE: ClassVar[bool] = False
    SUPPORTS_CHILDREN: ClassVar[bool] = False
    SUPPORTS_PARENTS: ClassVar[bool] = False

    # TB: am slightly uneasy of working around class hierarchy inversion this way,
    # but does definitely make preconditions and properties over Primitives much simpler
    @property
    def is_simple(self) -> bool:
        """Whether this Primitives is considered indivisible within its hierarchy"""
        # DEV: this is a mechanism to prevent Simples from being the parents of any
        # other Primitive without passing type info backward up the inheritance tree
        return self.IS_SIMPLE

    @property
    def supports_children(self) -> bool:
        """
        Whether this Primitive is capable of having other
        Primitives below it (i.e. at a "finer" resolution)
        in the MuPT molecular representation hierarchy
        """
        return self.SUPPORTS_CHILDREN

    @property
    def supports_parents(self) -> bool:
        """
        Whether this Primitive is capable of being bound as the child of
        another "parent" Primitive (which is at a "coarser" resolution)
        in the MuPT molecular representation hierarchy
        """
        return self.SUPPORTS_PARENTS

    # Instance attributes
    connections: ConnectorManager
    _shape: Optional[BoundedTransformableShape]  # TODO: add protected access
    _label: PrimitiveLabel
    _mass: Optional[float]
    _role: str
    metadata: Metadata

    _frozen_connections: bool
    _frozen_hierarchy: bool

    # Creation
    ## N.B.: deliberately not placing __init__ method here;
    ## should NOT be able to create instance of Primitive base
    ## TB: given that, may consider pivoting from Protocol -> ABC?
    def _init(
        self,
        # DEV: connectors deliberately omitted here; subtype each handle in unique way
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        mass: Optional[float] = None,
        role: str = "unassigned",
        metadata: Optional[Metadata] = None,
    ) -> None:
        """
        Boilerplate initialization of hidden attrs, with defaults
        To be called inside __init__ of concrete subtypes
        """
        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

        # instance attrs
        self._shape = shape
        self._label = label
        self._mass = mass
        self._role = role
        self.metadata = metadata or dict()

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

    @property  # TB: eventually, incorporate info about units
    def mass(self) -> Optional[float]:
        """The assigned molecular mass of this Primitive"""
        return self._mass

    # DEV: deliberately omitting setter; subtypes will impl particular procedures for it

    # Hierarchy
    ## Invariants
    # TB: in any hierarchy, key invariants that must be enforced at all times are:
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

    ## Traversal
    def hierarchy_tree(self, *args, **kwargs) -> DiGraph:
        """Generate a directed Graph representing the hierarchy below this Primitive"""
        return anytree_to_networkx(self, *args, **kwargs)

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

    ## Hierarchy mutability
    (
        frozen_hierarchy,
        _precondition_mutable_hierarchy,
        precondition_mutable_hierarchy,
    ) = precondition_factory(
        "_frozen_hierarchy",
        "Hierarchy of this Primitive is read-only accessible; "
        "no new or modified parent/child relationships allowed.",
    )

    ### Freeze
    def _freeze_hierarchy_local(self) -> None:
        """Prevent modification of parents/children of this Primitive ONLY"""
        self._frozen_hierarchy = True

    def _freeze_hierarchy_subsequent(self) -> None:
        """
        Prevent modification of parents/children of this
        Primitive and any below it in the hierarchy
        """
        self._freeze_hierarchy_local()
        for subprimitive in self.descendants:
            subprimitive._freeze_hierarchy_local()

    def freeze_hierarchy(self) -> None:
        """
        Prevent modification of parents/children of all Primitives in
        the hierarchy of this Primitive, even below AND above it
        """
        self.root._freeze_hierarchy_subsequent()

    ### Unfreeze
    def _unfreeze_hierarchy_local(self) -> None:
        """Enable modification of parents and children of this Primitive ONLY"""
        self._frozen_hierarchy = False

    def _unfreeze_hierarchy_subsequent(self) -> None:
        """
        Enable modification of parents/children of this
        Primitive and any below it in the hierarchy
        """
        self._unfreeze_hierarchy_local()
        for subprimitive in self.descendants:
            subprimitive._unfreeze_hierarchy_local()

    def unfreeze_hierarchy(self) -> None:
        """
        Enable modification of parents/children of all Primitives in
        the hierarchy of this Primitive, even below AND above it
        """
        self.root._unfreeze_hierarchy_subsequent()

    # Connectivity
    ## Reading
    @property
    def connectors(self) -> Collection[Connector]:
        """Convenience wrapper for accessing ALL connectors managed by this Primitive"""
        # TODO: also provide convenient access to connectors_free and connectors_bound
        return self.connections.connectors

    @property
    def functionality(self) -> int:
        """Number of free Connections this Primitive has access to"""
        return self.connections.functionality

    def fetch_connector(self, conn: ConnectorAddress | Connector) -> Connector:
        """Fetch a connector managed by this Priomitive, if it exists"""
        return self.connections.connector(connector_address_flexible(conn))

    ## Connector mutability
    (
        frozen_connections,
        _precondition_mutable_connectors,
        precondition_mutable_connectors,
    ) = precondition_factory(
        "_frozen_connections",
        "Connectors of this Primitive is read-only accessible; "
        "no new or modified neighbor relationships allowed.",
    )

    ### Freezing
    def _freeze_connections_local(self) -> None:
        """Prevent modification of Connectors on this Primitive ONLY"""
        self.connections = ConnectorManagerFrozen(*self.connections.connectors)
        self._frozen_connections = True

    def _freeze_connections_subsequent(self) -> None:
        """
        Prevent modification of Connectors managed by this
        Primitive and any below it in the hierarchy
        """
        self._freeze_connections_local()
        for subprimitive in self.descendants:
            subprimitive._freeze_connections_local()

    def freeze_connections(self) -> None:
        """
        Prevent modification of any Connectors on all Primitives in
        the hierarchy of this Primitive, even below AND above it
        """
        # TB: from root, since it doesn't make sense to just freeze parts of hierarchy;
        # if one bit is frozen, it causes all others touching it to also freeze
        # Note that "root" is not necessarily a RootPrimitive,
        # but rather the topmost Primitive ancestor in the current hierarchy
        self.root._freeze_connections_subsequent()

    ### Unfreezing
    def _unfreeze_connections_local(self) -> None:
        """Allow modification of Connectors on this Primitive ONLY"""
        self.connections = ConnectorManagerMutable(*self.connections.connectors)
        self._frozen_connections = False

    def _unfreeze_connections_subsequent(self) -> None:
        """
        Allow modification of Connectors managed by this
        Primitive and any below it in the hierarchy
        """
        self._unfreeze_connections_local()
        for subprimitive in self.descendants:
            subprimitive._unfreeze_connections_local()

    def unfreeze_connections(self) -> None:
        """
        Allow modification of any Connectors on all Primitives in
        the hierarchy of this Primitive, even below AND above it
        """
        # TB: Similarly, unfreeze from root down, since it doesn't
        # make sense to only have parts of hierarchy be mutable
        self.root._unfreeze_connections_subsequent()

    ## Modifying connectors
    @precondition_mutable_connectors
    def _add_connector(
        ## N.B.: deliberately private; only Simples should have explicit access
        self,
        connector: Connector,
        label: Optional[ConnectorLabelLike] = None,
    ) -> ConnectorAddress:
        """Add a new Connector to those managed locally"""
        self.connections.add_connector(connector, label=label)
        # N.B.: connector.holder deliberately not set here

        return connector.address

    @precondition_mutable_connectors
    def _remove_connector(
        self,
        connector_address: Connector | ConnectorAddress,
    ) -> Connector:
        """Remove an existing Connector from those managed locally"""
        # N.B.: connector neighbor deliberately untouched here
        return self.connections.remove_connector(connector_address)

    ## Neighborship
    def potential_neighbors_with_connectors(
        self,
    ) -> Generator[tuple[Connector, Connector, tuple["Primitive", ...]], None, None]:
        """
        Generate all potential neighbors, along with the
        Connectors linking them to *this* Primitive

        Yields
        ------
        our_conn, their_conn, nb_path : tuple[Connector, Connector, tuple[Primitive]]
            The pair of Connectors linking this Primitive to a branch of neighbor,
            and the branch of all (non-internal) neighbors at the other end, arranged
            by order in the hierarchy (i.e. least-deep first)
        """
        for our_connector in self.connections.connectors_bound:
            # any superprimitives which share connectors with the holder are
            # also considered neighbors; this is what enables multiscaling
            their_connector: Connector = our_connector.neighbor
            neighbor_leaf: Primitive = their_connector.holder
            # TB TODO: finesse typehints to suppress (perceived) unset NoneType values

            # avoid "internal" neighbors (of whom *this* Primitive is also a parent)
            if self in neighbor_leaf.path:
                continue

            common_ancestors: tuple[Primitive, ...] = commonancestors(
                self,
                neighbor_leaf,
            )
            neighbor_branch: tuple[Primitive, ...] = tuple(
                potential_neighbor
                for potential_neighbor in neighbor_leaf.path
                if potential_neighbor not in common_ancestors
            )

            yield our_connector, their_connector, neighbor_branch

    def potential_neighbors(self) -> Generator[tuple["Primitive", ...], None, None]:
        """Generate the non-internal neigbhor branches connected to this Primitive"""
        for _, _, neighbor_branch in self.potential_neighbors_with_connectors():
            yield neighbor_branch

    def neighbors_with_connectors(
        self,
        predicate: NodePredicate["Primitive"] = DEFAULT_PREDICATE,
    ) -> Generator[tuple[Connector, Connector, "Primitive"], None, None]:
        """
        Generates the two Connectors which constitute a connection
        to a neighboring Primitive, as well as that Primitive itself
        Yields as (our_connector, their_connector, them) tuples

        Can sub-select among resolutions of neighbor
        branches using a predicate, if provided
        """
        for (
            our_connector,
            their_connector,
            neighbor_branch,
        ) in self.potential_neighbors_with_connectors():
            # TB: will raise ValueError on empty branch
            # should never happen if holders are set correctly, but am
            # leaving note here to speed up debugging in case it ever does
            branch_root, *successors_list = neighbor_branch
            successors_iter = iter(successors_list)

            def successors(prim: "Primitive") -> list["Primitive"]:
                """
                Replacement for prim.children when stepping thru this branch

                Wrapper avoids StopIteration on shallow branches
                """
                next_prim: Optional["Primitive"] = next(successors_iter, None)
                if next_prim is None:
                    return []
                return [next_prim]

            for chosen_neighbor in primoprogenitors(
                branch_root, predicate=predicate, successors=successors
            ):
                yield our_connector, their_connector, chosen_neighbor

    def neighbors(
        self,
        predicate: NodePredicate["Primitive"] = DEFAULT_PREDICATE,
    ) -> Generator["Primitive", None, None]:
        """
        Primitives whose share a Connection with this one

        Can sub-select among resolutions of neighbor
        branches using a predicate, if provided
        """
        for _, _, neighbor in self.neighbors_with_connectors(predicate=predicate):
            yield neighbor

    def is_neighbors_with(self, other: "Primitive") -> bool:
        """
        Whether this Primitive is a neighbor of the other Primitive

        This relation is symmetric, i.e. a.is_neighbor_of(b) <=> b.is_neighbor_of(a)
        """
        for neighbor_branch in self.potential_neighbors():
            if other in neighbor_branch:
                return True
        else:
            return False

    def connect_neighbor(
        self,
        other: "Primitive",
        our_connector: Union[ConnectorAddress, Connector],
        their_connector: Union[ConnectorAddress, Connector],
        alignment_strategy: Optional[ConnectorAntialignmentStrategy] = None,
    ) -> None:
        """
        Forge a new connection to another Primitive

        If explicit Connectors are provided for either or both
        Primitives,will use those as halves of the connection;
        Otherwise, will attempt to deduce a unique choice using the linking algorithm
        """
        our_connector_chosen: Connector = self.fetch_connector(our_connector)
        their_connector_chosen: Connector = other.fetch_connector(their_connector)
        our_connector_chosen.neighbor = their_connector_chosen

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

    # Copying Primitives - implements Copyable contract on RigidlyTransformable
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
            ## TB TODO: SupportsChildren is more accurate annotation, but
            ## inverts inheritance hierarchy; find way to get best of both
            # new_parent: Optional[SupportsChildren] = orig_prim_to_new_prim.get(
            new_parent: Optional[Primitive] = orig_prim_to_new_prim.get(
                subprim.parent, None
            )
            clone_no_hierarchy_subprim.parent = new_parent
        own_copy: Self = orig_prim_to_new_prim[self]

        return own_copy, orig_prim_to_new_prim

    def _copy_untransformed(self) -> Self:
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
            # to be implemented on Simples going on - TB: slight inheritance inversion?
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

    # Geometry
    @property
    def shape(self) -> Optional[BoundedTransformableShape]:
        """The external shape of this Primitive"""
        return self._shape

    @shape.setter
    def shape(self, new_shape: Optional[BoundedTransformableShape]) -> None:
        """Assign a new shape to this Primitive"""
        if (new_shape is not None) and not isinstance(
            new_shape, BoundedTransformableShape
        ):
            raise TypeError(
                "Primitive.shape must be BoundedTransformableShape "
                f"or None, not {type(new_shape).__name__}"
            )
        self._shape = new_shape

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

    ## Implement RigidlyTransformable contract
    def _rigidly_transform(self, transformation: RigidTransform) -> None:
        """Apply a rigid transformation to all parts of a Primitive which support it"""
        self._rigidly_transform_connectors(transformation)
        self._rigidly_transform_shape(transformation)
        for subprimitive in self.descendants:  # descendants avoids recursive calls
            # N.B.: not transforming sub-primitives' Connectors since they are, in
            # aggregrate THE SAME Connectors managed here (don't double-transform)
            subprimitive._rigidly_transform_shape(transformation)

    # Depiction
    def __str__(self) -> str:
        """
        Output of calling str(...) on this Primitive

        Also the default representation of this Primitive
        when it is used as a node in any NetworkX graph
        """
        return f"{self.label!s}[{str(self.address)[: self.ADDRESS_PREVIEW_LEN]}]"

    # DEV: will likely have to change for subtypes
    # def __repr__(self) -> str:
    #     raise NotImplementedError

    def hierarchy_summary(
        self,
        to_depth: Optional[int] = None,
        style: Union[str, ConcreteStyle, type[ConcreteStyle]] = "round",
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
        ).by_attr(render_attr)


class SupportsChildren(Primitive):
    """
    Type of Primitive which is allowed to have
    other Primitives "beneath" it in a hierarchy.

    I.e. interpreting a representation hierarchy as a rooted tree,
    these Primitives are nodes which allow OUTGOING directed edges
    """

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

    @property
    def mass(self) -> Optional[float]:
        """The assigned molecular mass of this Primitive"""
        if self._mass is None:
            # DEV: will raise TypeError if ANY subprim has unset mass (intentional)
            self._mass = sum(child.mass for child in self.children)

        return self._mass

    def child(self, prim_addr: PrimitiveAddress) -> "SupportsParents":
        """
        Lookup a child Primitive by its address and
        return the Primitive instance, if present
        """
        return self.children_by_address[prim_addr]  # raise KeyError if not present

    fetch_primitive = child

    # Hierarchy management
    # TB: consider removing attach_detach child wrappers
    # and moving core logic to _pre_attach/_post_attach

    ## Attachment
    @Primitive.precondition_mutable_hierarchy(
        msg="Hierarchy modification is forbidden this Primitive; cannot attach"
    )
    @Primitive.precondition_mutable_connectors  # positions and neighbors may shift
    def _pre_attach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Preconditions prior to attempting to attach of this Primitive to a parent"""
        super()._pre_attach_children(children)

        for child in children:
            child._precondition_mutable_hierarchy()

    def _post_attach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Post-actions to take once children are attached and parent is bound"""
        super()._post_attach_children(children)
        # TODO: remap connection info
        ...

    @Primitive.precondition_mutable_hierarchy
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
        if label is not None:
            child.label = label
        self.children_by_address[child.address] = child

        return child.address

    ## Detachment
    @Primitive.precondition_mutable_hierarchy(
        msg="Hierarchy modification is forbidden this Primitive; cannot detach"
    )
    @Primitive.precondition_mutable_connectors  # positions and neighbors may shift
    def _pre_detach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Preconditions prior to attempting detachment of this Primitive from parent"""
        super()._pre_detach_children(children)

    def _post_detach_children(self, children: Iterable["SupportsParents"]) -> None:
        """Post-actions to take once children are detached and parent is unbound"""
        super()._post_detach_children(children)

    @Primitive.precondition_mutable_hierarchy
    def detach_child(self, prim_addr: PrimitiveAddress) -> Primitive:
        """
        Unregister an existing child Primitive, making it no
        be longer below this one in the resolution hierarchy
        """
        child = self.children_by_address.pop(prim_addr)
        child.parent = None

        return child


class SupportsParents(Primitive):
    """
    Type of Primitive which is allowed to have
    other Primitives "above" it in a hierarchy

    I.e. interpreting a representation hierarchy as a rooted tree,
    these Primitives are nodes which allow INCOMING directed edges
    """

    # Hierarchy
    # TB: you might be thinking it would be more natural to have checks on parent
    # Primitives in SupportParent instead; the reason for having them here instead is
    # setting children always calls `child.parent = new_parent_value` under the hood
    @Primitive.precondition_mutable_hierarchy
    def _pre_attach(self, parent: SupportsChildren) -> None:
        """Ensure both parent and child are fully mutable"""
        super()._pre_attach(parent)
        parent._precondition_mutable_hierarchy()

    def _post_attach(self, parent: SupportsChildren) -> None:
        """Once parent is set, inject own Connectors into hierarchy"""
        super()._post_attach(parent)
        for connector in self.connectors:
            self.inject_connector_into_hierarchy(connector)

    @Primitive.precondition_mutable_hierarchy
    def _pre_detach(self, parent: SupportsChildren) -> None:
        """Ensure both parent and child are fully mutable"""
        super()._pre_detach(parent)
        parent._precondition_mutable_hierarchy()

    def _post_detach(self, parent: SupportsChildren) -> None:
        """Remove all local connectors from the hierarchy above self"""
        super()._post_detach(parent)
        for connector in self.connectors:
            self.withdraw_connector_from_hierarchy(connector)

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
