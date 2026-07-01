'''Unit tests for containers module'''

import pytest

from typing import (
    Callable,
    ClassVar,
    Hashable,
    Iterable,
    Optional,
    TypeVar,
)
T = TypeVar('T')
from dataclasses import dataclass, field

from mupt.mutils.containers import UniqueRegistry, LabelT


@dataclass
class DummyRelation:
    DEFAULT_LABEL : ClassVar[str] = 'default'
    label : Hashable = field(default_factory=str)
    
# Concrete, initializable registry examples
def reg_example_a() -> UniqueRegistry:
    reg = UniqueRegistry()
    _ = reg.register_from({'letters' : 'ab', 'numbers' : (1,2,3)})

    return reg

def reg_example_b() -> UniqueRegistry:
    reg = UniqueRegistry()
    handles = reg.register_from({'letters' : 'bcd', 'truths' : (False, True)})

    return reg

def reg_example_c() -> UniqueRegistry:
    reg = UniqueRegistry()
    handles = reg.register_from([3.14, 0.5772, 2.718], label='constants')

    return reg


# Initialization tests
@pytest.mark.xfail(
    reason="Direct assignment to UniqueRegistry items should raise PermissionError",
    raises=PermissionError,
    strict=True,
)
def test_unique_reg_no_defaults() -> None:
    '''Test that key-value pairs cannot be directly intialized in UniqueRegistry'''
    reg = UniqueRegistry(this='is_illegal')

# Registration tests
def test_unique_reg_register_explicit_label() -> None:
    '''Test that registering with explicit label works as expected'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    reg.register(obj, label='my_label')
    
    assert set(reg.keys()) == {('my_label', 0)}

def test_unique_reg_register_implicit_label() -> None:
    '''Test that registering with implicit label (inferred from registered object) works as expected'''
    obj = DummyRelation(label='p')
    reg = UniqueRegistry()
    reg.register(obj)
    
    assert set(reg.keys()) == {('p', 0)}

@pytest.mark.parametrize(
    'collection,labeller,keys_expected',
    [
        # Test registration from mapping
        (
            {
                'letter' : 'abc',
                'number' : [1,2,3,4],
            },
            None,
            set([
                ('letter', 0),
                ('letter', 1),
                ('letter', 2),
                ('number', 0),
                ('number', 1),
                ('number', 2),
                ('number', 3),
            ]),
        ),
        (
            {
                'first' : (DummyRelation(label='p'),),
                'second' : (DummyRelation(label='q'), DummyRelation(label='p')),
            },
            None,
            set([ # explicit label overrides object label
                ('first', 0),
                ('second', 0),
                ('second', 1),
            ]),          
        ),
        # Testing registration from labelled objects
        (
            [5,6,7,8],
            'begin',
            set([
                ('begin', 0),
                ('begin', 1),
                ('begin', 2),
                ('begin', 3),
            ]),
        ),
        # Testing registration with explicit base label
        (
            (
                DummyRelation(label='p'),
                DummyRelation(label='q'),
                DummyRelation(label='q'),
                DummyRelation(label='r'),
            ),
            None,
            set([
                ('p', 0),
                ('q', 0),
                ('q', 1),
                ('r', 0),
            ]),
        ),
        # Test registration with Callable label generator
        (
            [
                'foo',
                'bar',
                'baz',
            ],
            str.swapcase,
            set([
                ('FOO', 0),
                ('BAR', 0),
                ('BAZ', 0),
            ]),
        ),
    ]
)
def test_register_from(
    collection : Iterable[T],
    labeller : Optional[Callable[[T], LabelT] | LabelT],
    keys_expected : set[tuple[LabelT, int]],
) -> None:
    '''Check that bulk registration behaves as expected'''
    reg = UniqueRegistry()
    keys_actual = reg.register_from(collection, label=labeller)

    assert set(keys_actual) == keys_expected

# Deregistration tests
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

@pytest.mark.parametrize(
    'reg',
    (    
        reg_example_a(),
        reg_example_b(),
        reg_example_c(),
    ),
)
def test_reset_ticker_total(reg : UniqueRegistry) -> None:
    '''
    Test that resetting running ticker counts for ALL keys sets all counts to 0
    '''
    orig_keys = set(reg._ticker.keys()) # wrap in new container to prevent any chance of accidentally referencing original
    reg.reset_ticker()
    assert all(reg._ticker[key] == 0
        for key in orig_keys
    )
    
@pytest.mark.parametrize(
    'reg,key',
    [
        ...
    ]
)
def test_reset_ticker_indiv(reg : UniqueRegistry, key : LabelT) -> None:
    '''
    Test that resetting running ticker counts for a specific
    key sets that count (and ONLY that count) to 0
    '''
    orig_ticker : dict[LabelT, int] = dict(reg._ticker)
    other_keys = set(reg._ticker.keys()) - {key}

    reg.reset_ticker_count_for(key)
    assert reg._ticker[key] == 0 \
        and all(reg._ticker[other_key] == orig_ticker[other_key]
            for other_key in other_keys
        )

# Internal state update tests
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
    
# Copying tests
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

# Partition tests
@pytest.mark.parametrize(
    'reg1,reg2,dict_expected',
    [
        (
            reg_example_a(),
            reg_example_b(),
            {
                ('letters', 0) : 'a',
                ('letters', 1) : 'b',
                ('numbers', 0) : 1,
                ('numbers', 1) : 2,
                ('numbers', 2) : 3,
                ('letters', 2) : 'b',
                ('letters', 3) : 'c',
                ('letters', 4) : 'd',
                ('truths', 0) : False,
                ('truths', 1) : True,
            }
        )
    ]
)
def test_merge(
    reg1 : UniqueRegistry,
    reg2 : UniqueRegistry,
    dict_expected : dict[tuple[LabelT, int], str | int | bool],
) -> None:
    key_remap = reg1.merge(reg2)
    assert dict(reg1) == dict_expected

@pytest.mark.parametrize(
    'regs,dict_expected',
    [
        (
            (    
                reg_example_a(),
                reg_example_b(),
                reg_example_c(),
            ),
            {
                ('letters', 0) : 'a',
                ('letters', 1) : 'b',
                ('numbers', 0) : 1,
                ('numbers', 1) : 2,
                ('numbers', 2) : 3,
                ('letters', 2) : 'b',
                ('letters', 3) : 'c',
                ('letters', 4) : 'd',
                ('truths', 0) : False,
                ('truths', 1) : True,
                ('constants', 0) : 3.14,
                ('constants', 1) : 0.5772,
                ('constants', 2) : 2.718,
            }
        ),
    ]
)
def test_merged(
    regs : Iterable[UniqueRegistry],
    dict_expected : dict[tuple[LabelT, int], str | int | bool],
) -> None:
    '''Test that classmethod version of merge() behaves as expected'''
    reg, handle_maps = UniqueRegistry.merged(*regs)
    assert dict(reg) == dict_expected

def test_split() -> None:
    '''Test that splitting by category works as expected'''
    reg = UniqueRegistry()
    _ = reg.register_from(range(9), 'num')

    subregs = reg.split(lambda x : x % 3)
    subregs = {category : dict(subreg) for category, subreg in subregs.items()} # conversion done purely for easy comparison

    assert subregs == {
        0 : {('num', 0) : 0, ('num', 1) : 3, ('num', 2) : 6},
        1 : {('num', 0) : 1, ('num', 1) : 4, ('num', 2) : 7},
        2 : {('num', 0) : 2, ('num', 1) : 5, ('num', 2) : 8},
    }
