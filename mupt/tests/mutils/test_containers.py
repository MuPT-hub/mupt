'''Unit tests for containers module'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

from typing import ClassVar, Hashable
from dataclasses import dataclass, field

from mupt.mutils.containers import UniqueRegistry


@dataclass
class DummyRelation:
    DEFAULT_LABEL : ClassVar[str] = 'default'
    label : Hashable = field(default_factory=str)
    
# initialization tests
@pytest.mark.xfail(
    reason="Direct assignment to UniqueRegistry items should raise PermissionError",
    raises=PermissionError,
    strict=True,
)
def test_unique_reg_no_defaults() -> None:
    '''Test that key-value pairs cannot be directly intialized in UniqueRegistry'''
    reg = UniqueRegistry(this='is_illegal')

# registration tests
def test_unique_reg_register_explicit() -> None:
    '''Test that registering with explicit label works as expected'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    reg.register(obj, label='my_label')
    
    assert set(reg.keys()) == {('my_label', 0)}

def test_unique_reg_register_implicit() -> None:
    '''Test that registering with implicit label (inferred from registered object) works as expected'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    
    assert set(reg.keys()) == {('p', 0)}
    
def test_unique_reg_register_from_explicit() -> None:
    '''Test that registering multiple objects with explicit labels works as expected'''
    obj1 = DummyRelation(label='p')
    obj2 = DummyRelation(label='q')
    reg = UniqueRegistry()
    handles = reg.register_from({'first' : obj1, 'second' : obj2})
    
    assert set(reg.keys()) == {('first', 0), ('second', 0)}
    
def test_unique_reg_register_from_implicit() -> None:
    '''Test that registering multiple objects with implicit labels (inferred from registered objects) works as expected'''
    obj1 = DummyRelation(label='p')
    obj2 = DummyRelation(label='q')
    reg = UniqueRegistry()
    handles = reg.register_from([obj1, obj2])
    
    assert set(reg.keys()) == {('p', 0), ('q', 0)}

# deregistration tests
def test_unique_reg_deregister() -> None:
    '''Test that deregistering an item removes it from the registry and returns the object'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    handle = reg.register(obj)
    removed_obj = reg.deregister(handle)
    
    assert (removed_obj == obj) and (len(reg) == 0)
    
def test_unique_reg_subscript() -> None:
    '''Test that unique registry items can be accessed via subscript notation'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    
    assert reg[('p', 0)] == obj
    
def test_unique_reg_deletion() -> None:
    '''Test that unique registry items can be deleted via del operator'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    del reg[('p', 0)]
    
    assert len(reg) == 0
    
def test_unique_reg_purge() -> None:
    '''Test that purging a label removes all associated objects'''
    reg = UniqueRegistry()
    a = DummyRelation(label='a')
    for _ in range(3):
        reg.register(a)
    
    b = DummyRelation(label='b')
    for _ in range(4):
        reg.register(b)
        
    reg.purge('a')
    assert all(handle[0] != 'a' for handle in reg.keys()) and (len(reg) == 4)

# internal state update tests
def test_freed_labels_reinserted() -> None:
    '''Test that freed unique indices are reused upon reinsertion before continuing to use incremented labels'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    for _ in range(4):
        reg.register(obj)
        
    _ = reg.deregister(('p', 1))
    _ = reg.deregister(('p', 2))
    reg.register(obj)
    
    assert set(reg.keys()) == {('p', 0), ('p', 1), ('p', 3)}
    
def test_unique_reg_adjust_ticker() -> None:
    '''Test that the ticker count adjustments shift uniquifying index accordingly'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    
    reg.register(obj)
    reg.adjust_ticker_count_for('p', 5)
    reg.register(obj)
    
    assert set(reg.keys()) == {('p', 0), ('p', 5)}
    
# copying tests
def test_unique_reg_copy() -> None:
    '''Test that copying a UniqueRegistry produces an identical copy'''
    reg = UniqueRegistry()
    a = DummyRelation(label='a')
    b = DummyRelation(label='b')
    
    reg.register(a)
    reg.register(b)
    copy_reg = reg.copy()
    
    assert (
        reg.keys() == copy_reg.keys()
        and reg._ticker == copy_reg._ticker
        and reg._freed == copy_reg._freed
    )
    
def test_unique_reg_copy_ticker_indep() -> None:
    '''Test that the ticker state of a copied UniqueRegistry is independent of the ticker of the original'''
    reg = UniqueRegistry()
    a = DummyRelation(label='a')
    
    reg.register(a)
    copy_reg = reg.copy()
    reg.register(a) # ought to have no effect on copy
    
    assert copy_reg._ticker != reg._ticker
    
def test_unique_reg_copy_freed_indep() -> None:
    '''Test that the freed labels state of a copied UniqueRegistry is independent of the original'''
    reg = UniqueRegistry()
    a = DummyRelation(label='a')

    reg.register(a)
    copy_reg = reg.copy()
    reg.deregister(('a', 0)) # ought to have no effect on copy

    assert copy_reg._freed != reg._freed

# Label access tests
def test_unique_reg_by_labels() -> None:
    '''Test that by_labels property returns correct mapping from labels to tuples of registered objects'''
    reg = UniqueRegistry()
    a = DummyRelation(label='a')
    b = DummyRelation(label='b')
    c = DummyRelation(label='c')
    
    reg.register(a)
    reg.register(b)
    reg.register(c, label='a')
    
    by_labels = reg.by_labels
    assert set(by_labels.keys()) == {'a', 'b'}
    assert by_labels['a'] == (a, c)
    assert by_labels['b'] == (b,)