'''Unit tests for object referencing'''

from typing import Any, Type
from dataclasses import dataclass
from uuid import UUID

import pytest
from mupt.mutils.referencing import Addressable


# dummy Addressable classes
class DummyNoArgs(Addressable):
    ...

class DummyWithArgs(Addressable):
    def __init__(self, foo : str, bar : int=123) -> None:
        self.foo = foo
        self.bar = bar

@dataclass # want to ensure addr registration mechanisms plays nice w/ dataclasses
class DummyDataclass(Addressable):
    baz : str
    boo : float


# tests proper
@pytest.mark.parametrize(
    'addr_typ,args',
    [
        (DummyNoArgs, {}),
        (DummyWithArgs, {'foo' : 'abc', 'bar' : 42}),
        (DummyDataclass, {'baz' : 'name', 'boo' : 3.14}),
    ]
)
def test_has_address(addr_typ : Type[Addressable], args : dict[str, Any]) -> None:
    '''Test that Addressable objects indeed implement the address they claim to'''
    obj = addr_typ(**args)
    assert hasattr(obj, 'address')
    assert hasattr(obj, '_uuid') and isinstance(obj._uuid, UUID)

def test_address_registration() -> None:
    '''Test that newly-minted objects are also registered by their address in the classwide registry'''
    ...

def test_weak_address_refs() -> None:
    '''Test that records of objects in classwide registry automatically vanish when object is garbage collected'''
    class DummyLocal(Addressable):
        ... # N.B.: defined locally to ensure reference counter to instances is not contaminated by other tests

    obj = DummyLocal()
    assert len(DummyLocal.registry_addresses) == 1

    del obj 
    assert len(DummyLocal.registry_addresses) == 0

def test_object_registries_distinct() -> None:
    '''Test that distinct subtypes of Addressable do not share their classwide object registries'''
    ...