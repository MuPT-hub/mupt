'''Unit tests for containers module'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

from typing import ClassVar, Hashable
from dataclasses import dataclass, field

from mupt.mutils.containers import UniqueRegistry


@dataclass
class TestObj:
    DEFAULT_LABEL : ClassVar[str] = 'default'
    label : Hashable = field(default_factory=str)
    
    
@pytest.mark.xfail(
    reason="Direct assignment to UniqueRegistry items should raise PermissionError",
    raises=PermissionError,
    strict=True,
)
def test_unique_reg_no_defaults() -> None:
    '''Test that key-value pairs cannot be directly intialized in UniqueRegistry'''
    reg = UniqueRegistry(this='is_illegal')

def test_unique_reg_register_explicit() -> None:
    '''Test that registering with explicit label works as expected'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    reg.register(obj, label='my_label')
    assert set(reg.keys()) == {('my_label', 0)}

def test_unique_reg_register_implicit() -> None:
    '''Test that registering with implicit label (inferred from registered object) works as expected'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    assert set(reg.keys()) == {('p', 0)}
    
def test_unique_reg_register_from_explicit() -> None:
    '''Test that registering multiple objects with explicit labels works as expected'''
    obj1 = TestObj(label='p')
    obj2 = TestObj(label='q')
    reg = UniqueRegistry()
    reg.register_from({'first' : obj1, 'second' : obj2})
    assert set(reg.keys()) == {('first', 0), ('second', 0)}
    
def test_unique_reg_register_from_implicit() -> None:
    '''Test that registering multiple objects with implicit labels (inferred from registered objects) works as expected'''
    obj1 = TestObj(label='p')
    obj2 = TestObj(label='q')
    reg = UniqueRegistry()
    reg.register_from([obj1, obj2])
    assert set(reg.keys()) == {('p', 0), ('q', 0)}

def test_unique_reg_unregister() -> None:
    '''Test that unregistering an item removes it from the registry and returns the object'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    handle = reg.register(obj)
    removed_obj = reg.unregister(handle)
    assert (removed_obj == obj) and (len(reg) == 0)
    
def test_unique_reg_subscript() -> None:
    '''Test that unique registry items can be accessed via subscript notation'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    assert reg[('p', 0)] == obj
    
def test_unique_rreg_deletion() -> None:
    '''Test that unique registry items can be deleted via del operator'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    del reg[('p', 0)]
    assert len(reg) == 0

def test_freed_labels_reinserted() -> None:
    '''Test that freed unique indices are reused upon reinsertion before continuing to use incremented labels'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    for _ in range(4):
        reg.register(obj)
        
    _ = reg.unregister(('p', 1))
    _ = reg.unregister(('p', 2))
    reg.register(obj)
    
    assert set(reg.keys()) == {('p', 0), ('p', 1), ('p', 3)}
    
def test_unique_reg_adjust_ticker() -> None:
    '''Test that the ticker count adjustments shift uniquifying index accordingly'''
    obj = TestObj(label='p')
    reg = UniqueRegistry()
    
    reg.register(obj)
    reg.adjust_ticker_count_for('p', 5)
    reg.register(obj)
    
    assert set(reg.keys()) == {('p', 0), ('p', 5)}
    
def test_unique_reg_purge() -> None:
    '''Test that purging a label removes all associated objects'''
    reg = UniqueRegistry()
    a = TestObj(label='a')
    for _ in range(3):
        reg.register(a)
    
    b = TestObj(label='b')
    for _ in range(4):
        reg.register(b)
        
    reg.purge('a')
    assert all(handle[0] != 'a' for handle in reg.keys()) and (len(reg) == 4)