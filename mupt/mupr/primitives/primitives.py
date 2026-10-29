"""
Base classes for Primitives, the fundamental datatype
within MuPT's multiscale molecular representation
"""

from typing import (
    Any,
    ClassVar,
    Collection,
    Hashable,
    Optional,
    Self,
    Union,
)

type PrimitiveLabel = Hashable
type PrimitiveAddress = Hashable

from anytree.node import NodeMixin
from anytree.iterators import LevelOrderIter
from anytree.search import findall
from anytree.render import RenderTree

from networkx.classes import DiGraph

from scipy.spatial.transform import RigidTransform

from ..connection.types import ConnectorAddress
from ..connection.connectors import Connector
from ..connection.management import ConnectorManager, connector_address_flexible
from ..connection.alignment import ConnectorAntialignmentBallistic

from ...mutils.referencing import Addressed
from ...mutils.containers import Labelled
from ...geometry.shapes import Shaped, BoundedTransformableShape
from ...geometry.transforms.rigid import RigidlyTransformable
from ...trees.digraph import anytree_to_networkx
from ...trees.subselect import NodePredicate
from ...trees.render import tree_render_style, ConcreteStyle


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
    metadata: dict[Hashable, Any]

    _frozen_connections: bool
    _frozen_hierarchy: bool

    # Creation
    ## N.B.: deliberately not placeing __init__ method here;
    ## should NOT be able to create instance of Primitive base
    ## TB: given that, may consider pivoting from Protocol -> ABC?
    def _init(
        self,
        # DEV: connectors deliberately omitted here; subtype each handle in unique way
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        metadata: Optional[dict] = None,
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
        self.metadata = metadata or dict()
        self._label = label

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

    # Wrapped properties
    @property
    def connectors(self) -> Collection[Connector]:
        """Convenience wrapper for accessing ALL connectors managed by this Primitive"""
        # TODO: also provide convenient access to connectors_free and connectors_bound
        return self.connections.connectors

    @property
    def functionality(self) -> int:
        """Number of free Connections this Primitive has access to"""
        return self.connections.functionality

    # Copying
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
        own_copy: Self = orig_prim_to_new_prim[self]

        for subprim in LevelOrderIter(self):
            clone_no_hierarchy_subprim = subprim._copy_instance()
            orig_prim_to_new_prim[subprim] = clone_no_hierarchy_subprim

            # new_parent: Optional[SupportsChildren] = orig_prim_to_new_prim.get(
            new_parent: Optional[Primitive] = orig_prim_to_new_prim.get(
                subprim.parent, None
            )
            clone_no_hierarchy_subprim.parent = new_parent

        return own_copy, orig_prim_to_new_prim

    ## implement Copyable contract on RigidlyTransformable
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

    # Hierarchy
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

    def hierarchy_tree(self, *args, **kwargs) -> DiGraph:
        """Generate a directed Graph representing the hierarchy below this Primitive"""
        return anytree_to_networkx(self, *args, **kwargs)

    # Topology
    def fetch_connector(self, conn: ConnectorAddress | Connector) -> Connector:
        """Fetch a connector managed by this Priomitive, if it exists"""
        return self.connections.connector(connector_address_flexible(conn))

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
