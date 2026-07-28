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

def addressable_object_examples() -> tuple[tuple[Type[Addressable], dict[str, Any]], ...]:
    '''
    Pre-packaged examples of Addressable types and valid
    arguments needed to initialize an instance of those types
    '''
    return (
        (DummyNoArgs, {}),
        (DummyWithArgs, {'foo' : 'abc', 'bar' : 42}),
        (DummyDataclass, {'baz' : 'name', 'boo' : 3.14}),
    ) 

# tests proper
@pytest.mark.parametrize('addr_typ,kwargs', addressable_object_examples())
def test_has_address(addr_typ : Type[Addressable], kwargs : dict[str, Any]) -> None:
    '''Test that Addressable objects indeed implement the address they claim to'''
    obj = addr_typ(**kwargs)
    assert hasattr(obj, 'address')
    assert hasattr(obj, '_uuid') and isinstance(obj._uuid, UUID)

@pytest.mark.parametrize('addr_typ,kwargs', addressable_object_examples())
def test_address_registration(addr_typ : Type[Addressable], kwargs : dict[str, Any]) -> None:
    '''Test that newly-minted objects are also registered by their address in the classwide registry'''
    num_obj_registered_init : int = len(addr_typ.registry_addresses)
    obj = addr_typ(**kwargs)

    assert (obj.address in addr_typ.registry_addresses)  \
        and (len(addr_typ.registry_addresses) == (num_obj_registered_init + 1))

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
    class DummyLocal1(Addressable):
        ...

    class DummyLocal2(Addressable):
        ...

    obj1 = DummyLocal1()
    assert (obj1.address not in DummyLocal2.registry_addresses)

    obj2 = DummyLocal2() # perform reciprocal test to check symmetry
    assert (obj2.address not in DummyLocal1.registry_addresses)