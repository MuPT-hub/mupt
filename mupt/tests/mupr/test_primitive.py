"""Unit tests for Primitive interactions with one another and with sub-components"""

import pytest

from mupt.mupr.primitives import (
    Primitive,
    PrimitiveAddress,
    RootPrimitive,
    CompositePrimitive,
    SimplePrimitive
)


# Settings neighbors and topologies
def test_connect_neighbor():
    ...
    
def test_neighborship_propagates_thru_hierarchy():
    ...
    
    
# Sub-selecting Primitives
def test_neighbors_unconditional():
    ...
    
def test_neighbors_subset():
    ...

def test_primitive_predicates():
    ...
    
    
def test_cross_section():
    ...


# Inserting Connectors into and deleting Connectors from from hierarchy
def test_simple_add_connector():
    ...
    
def test_simple_remove_connector():
    ...
    
def test_simple_inject_connector_into_hierarchy():
    ...

def test_simple_withdraw_connector_from_hierarchy():
    ...
    