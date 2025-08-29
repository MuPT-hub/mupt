'''Generic Protocols for copyable objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Protocol, Self


class Copyable(Protocol):
    '''Any class which supports creating a copy of instances of the class'''
    def copy(self) -> Self:
        ...
