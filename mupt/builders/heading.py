'''Markers for indicating the direction of connections, paths, and other oriented traversable objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from enum import Enum, StrEnum, auto


class TerminalGroup(StrEnum):
    '''For indicating orientation of terminal monomers in a polymer chain'''
    HEAD = 'head'
    TAIL = 'tail'

class TraversalDirection(Enum):
    '''
    Uniquifying label indicating whether a connection faces "forward" or "backward" along a path graph 
    relative to an arbitrary-but-consistent absolute direction of traversal along the path from end-to-end
    '''
    AMBI = 0
    ANTERO = 1
    RETRO = 2
    
    @classmethod
    def complement(cls, direction : 'TraversalDirection') -> 'TraversalDirection':
        '''
        Get the complement (i.e. "opposite") direction to a given TraversalDirection
        
        Parameters
        ----------
        direction : TraversalDirection
            The direction to get the complement of
            
        Returns
        -------
        TraversalDirection
            The complement of the given direction
        '''
        if direction == cls.ANTERO:
            return cls.RETRO
        elif direction == cls.RETRO:
            return cls.ANTERO
        elif direction == cls.AMBI:
            return cls.AMBI