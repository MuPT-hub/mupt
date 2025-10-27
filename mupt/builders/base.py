'''Generic, base contract for all MuPT builders'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Iterable
from abc import ABC, abstractmethod

from scipy.spatial.transform import Rotation, RigidTransform

from ..mupr.primitives import Primitive, PrimitiveHandle


class PlacementGenerator(ABC):
    '''
    Abstract base class for all MuPT system builders
    
    Defines interface for generating centers & orientations (referred to as "placements" here),
    represented by RigidTransform instances, for all children of a given Primitive
    '''
    @abstractmethod
    def __init__(self) -> None:
        '''
        Implementation-specific parameters (e.g. force constants, target bond lengths, etc.) should be bound here
        '''
        ...
    
    def check_preconditions(
        self,
        primitive : Primitive,
    ) -> None:
        # DEV: deliberately NOT abstract, as some builders may opt to simply not check anything (establish this as the default behavior)
        '''
        Check that the given Primitive meets any preconditions for this builder
        E.g. a linear-only chain builder might check that a system's topology has no branches
        
        Implementation provided (if any) should raise detailed Exceptions if 
        preconditions are not met and pass without Exception/return otherwise
        '''
        ...
    
    @abstractmethod
    def _generate_placements(
        self,
        primitive : Primitive,
    ) -> Iterable[tuple[PrimitiveHandle, RigidTransform]]:
        '''Implement generation of rigid transformations to place each child of the given Primitive here'''
        ...
        
    def generate_placements(
        self,
        primitive : Primitive,
    ) -> Iterable[tuple[PrimitiveHandle, RigidTransform]]:
        '''
        Accepts a Primitive (with associated topology, connections, and hierarchy of children below it)
        and should generate placements (as RigidTransform instances) for all children below the provided Primitive,
        identified by the handles of each child Primitive
        '''
        self.check_preconditions(primitive)
        yield from self._generate_placements(primitive)

