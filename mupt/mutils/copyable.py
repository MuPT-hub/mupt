'''Generic Protocols for copyable objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Protocol, Self, runtime_checkable
from functools import cached_property


def clear_cached_properties(obj : object) -> None:
    '''
    Invalidate all cached_property values on an instance which are:
    * have been set on the instance
    * are defined in the corresponding type
    '''
    typ = type(obj)
    for method_name in dir(typ):
        methodlike = getattr(typ, method_name)
        if isinstance(methodlike, cached_property) and (method_name in vars(obj)):
            delattr(obj, method_name)

@runtime_checkable
class Copyable(Protocol):
    '''Any class which supports creating a copy of instances of the class'''
    def copy(self) -> Self:
        ...

class NotCopyableError(NotImplementedError):
    '''Raised when a copy-based operation is invokes on an object whose class doesn't implement it'''
    ...