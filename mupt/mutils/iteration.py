'''Tools for simplifying iteration over collections of items'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Sequence
from itertools import count


def ad_infinitum(value : Any) -> Generator[Any, None, None]:
    '''Wrap a single value in an inexhaustible generator which always returns that value'''
    while True:
        yield value
        
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