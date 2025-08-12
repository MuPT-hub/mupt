'''Defines interfaces and Protocols for types of object comparison among MuPT core objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Hashable, Iterable, Protocol, Self, TypeVar
T = TypeVar('T')
from collections import defaultdict


class Comparable(Protocol):
    '''Objects which can be compared by spatial similarity (coincidence), structural similarity (congruence), or both'''
    def coincident_with(self, other: Self) -> bool:
        ...
        
    def congruent_to(self, other: Self) -> bool:
        ...

    def equivalent_to(self, other: Self) -> bool:
        ...

def equivalence_classes(objects : Iterable[T], by_property : Callable[[T], Hashable]) -> set[frozenset[T]]:
    '''Generate equivalence classes of objects of type T according to a binary relation'''
    equiv_classes = defaultdict(set)
    for obj in objects:
        equiv_classes[by_property(obj)].add(obj)

    return set(frozenset(equiv_class) for equiv_class in equiv_classes.values())