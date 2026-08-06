'''Generic Protocols for copyable objects'''


from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class Copyable(Protocol):
    '''Any class which supports creating a copy of instances of the class'''
    def copy(self) -> Self:
        ...

class NotCopyableError(Exception):
    '''Raised when a copy-based operation is invokes on an object whose class doesn't implement it'''
    ...