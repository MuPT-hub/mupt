'''Interface for defining the internal structure of molecular Primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Generator,
    Generic,
    Iterable,
    TypeVar,
    Protocol,
    runtime_checkable,
)
from abc import ABC, abstractmethod

from .canonicalize import Canonicalizable
C = TypeVar('C', bound=Canonicalizable) 


@runtime_checkable
class Composite(Protocol):
    '''Object which contains some other internal components'''
    def components(self) -> Generator[C, None, None]:
        '''Iterate over all components contained by this Composite'''
        ...

class Structure(ABC, Generic[C]):
    '''Interface for defining the internal structure of a Primitive'''
    @property # TODO: make this more generic (no specific notion of atoms at this point)
    @abstractmethod
    def num_atoms(self) -> int:
        '''Number of atoms collectively held within the structure'''
        ...
        
    @property
    @abstractmethod
    def is_composite(self) -> bool:
        '''Whether the Structure is composite, i.e. contains other ??? as components'''
        ...
        
    @abstractmethod
    def _get_components(self) -> Iterable[C]:
        '''Implement how a particular type of Structure accesses and returns its successors'''
        ...
        
    def components(self) -> Generator[C, None, None]:
        '''Iterate over all components contained by this Structure'''
        for component in self._get_components():
            yield component
            if isinstance(component, Composite):
                yield from component.components()
        
    @abstractmethod
    def canonical_form(self) -> str: # DEVNOTE: this ensures Structure is itself Canonicalizable as well
        '''Return a canonical form used to distinguishing equivalent Structures'''
        ...
        
    def __hash__(self):
        return hash(self.canonical_form()) # DEVNOTE: guaranteed to be well-defined, since the canonical form returned is Hashable
        
class DiscreteStructure(Structure[C]):
    '''
    A Structure which contains a definite number of components without well-defined connectivity among them
    I.e. each singleton is distinct from every other, as in a discrete topology
    '''
    def __init__(self, components: Iterable[C]) -> None:
        self._components = list(components)
    
    @property
    def num_atoms(self) -> int:
        return sum(component.num_atoms for component in self._components)
    
    @property
    def is_composite(self) -> bool:
        return True
    
    def _get_components(self) -> Iterable[C]:
        return iter(self._components)
    
    def canonical_form(self) -> str:
        return '|'.join(component.canonical_form() for component in self._components)

# TODO: CardinalStructure, which has well-defined number of components, but not a form for them