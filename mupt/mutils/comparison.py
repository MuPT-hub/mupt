'''Defines interfaces and Protocols for types of object comparison among MuPT core objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Hashable, Iterable, Protocol, Self, TypeVar
T = TypeVar('T')


class Comparable(Protocol):
    '''Objects which can be compared by spatial similarity (coincidence), structural similarity (congruence), or both'''
    def coincides_with(self, other: Self) -> bool:
        '''Whether the parts of this object are spatially identical to those of the other'''
        ...
        
    def resembles(self, other: Self) -> bool: # DEV: resemblant_of()??
        '''Whether the parts of this object are of the same types, without necessarily being coincident or identical objects'''
        ...

    def congruent_to(self, other: Self) -> bool:
        '''Not necessarily coincident, but capable of becoming coincident (e.g. by orthogonal Procrustes)'''
        ...
        
    def fungible_with(self, other: Self) -> bool: # NOTE: opted not to use "interchangeable" to avoid confusion w/ OpenFF Interchange
        '''Whether this object can be replaced by "other" without affecting the behavior or output of programs involving this object'''
        return self.coincides_with(other) and self.resembles(other)

    # DEV: other terms to consider: "equivalent", "similar", "analogous", ...
    