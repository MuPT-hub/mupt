'''Tools for simplifying iteration over collections of items'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Any,
    Generator,
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
    Union,
)
T = TypeVar('T')

from collections import deque
from itertools import count, islice


def iter_len(itera : Iterable[T]) -> int:
    '''
    Get number of elements in an iterable object, even if unsized (namely a generator)
    
    Note that this will "use up" an iterator on call, i.e. 
    DON'T call this on collections you intend to iterate over later
    '''
    return sum(1 for _  in itera)

def ad_infinitum(value : Any) -> Generator[Any, None, None]:
    '''Wrap a single value in an inexhaustible generator which always returns that value'''
    while True:
        yield value
        
def flexible_iterator(
    values : Union[T, Iterable[T], Generator[T, None, None]],
    allowed_types : Union[tuple[type, ...], Union[type]],
) -> Iterator[T]:
    '''
    Create an iterator from a variety of input types
    Simplifies input to callables which expect iterators
    
    Behavior depends on the type of "values", namely:
    * Generator: returned as-is
    * Iterable: return as iterator
    * Single value: an infinite stream of that value if it is one of the allowed types, or Exception otherwise
    '''
    if isinstance(values, Generator):
        return values
    elif isinstance(values, Iterable):
        return iter(values)
    elif isinstance(values, allowed_types):
        return ad_infinitum(values)
    else:
        raise TypeError(f'Singleton values converted to iterator must be an instance of {allowed_types}, not {type(values)}')
        
def sliding_window(items : Iterable[T], n : int=1) -> Generator[tuple[T], None, None]:
    '''Generates sliding windows of width n over an iterable collection of items
    E.g. : sliding_window('ABCDE', 3) --> (A, B, C), (B, C, D), (C, D, E)
    '''
    it = iter(items)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)

    for x in it: # implicit else
        window.append(x) # owing to maxlen constraint, the first item in the is automatically discarded
        yield tuple(window)        

def int_complement(integers : Sequence[int], bounded : bool=False) -> Generator[int, None, None]:
    '''
    Given a sequence of integers, generates the complement of that sequence within the natural numbers,
    i.e. all non-negative integers which don't appear in that sequence, in ascending order
    
    By default, has no upper limit and will continue to generate integers indefinitely;
    however, generation can be capped at the maximum of the sequence by setting `bounded=True`
    
    Parameters
    ----------
    integers : Sequence[int]
        A sequence of integers to exclude from the natural numbers
    bounded : bool, default False
        Whether to limit enumeration to the maximum of the provided integer sequence
        
    Returns
    -------
    complement : Generator[int, None, None]
        A generator yielding all integers not present in the provided sequence, in ascending order
    '''
    _max = max(integers) # cache maximum (precludes use of generator-like sequence)
    for i in range(_max): # TODO: include choice for minimum?
        if i not in integers:
            yield i

    if not bounded: # keep counting past max if unbounded
        yield from count(start=_max + 1, step=1)