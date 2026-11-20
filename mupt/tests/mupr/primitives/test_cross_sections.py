"""
Unit tests for extracting predicate-determined cross
section graphs from a MuPT representation hierarchy

Cross-section represents a system at a given level of resolution
"""

import pytest

from mupt.mupr.primitives.primitive import Primitive, SupportsChildren
from mupt.trees.subselect import NodePredicate


@pytest.mark.parametrize(
    "root,predicate",
    [],
)
def test_cross_section_nodes(
    root: SupportsChildren,
    predicate: NodePredicate[Primitive],
) -> None:
    """
    Test that the Primitives primoprogenitors
    selected for by a chosen predicate are made
    the nodes of the cross-section graph
    """
    ...


@pytest.mark.parametrize(
    "root,predicate",
    [],
)
def test_cross_section_edges(
    root: SupportsChildren,
    predicate: NodePredicate[Primitive],
) -> None:
    """
    Test that neighboring Primitives in a cross
    section graph are each spanned by an edge
    """
    ...


def test_cross_section_bond_orders():
    """
    Test that bond order info is completely transferred
    to the edges of the cross-section graph
    """
    ...
