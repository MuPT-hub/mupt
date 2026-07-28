'''Utilities for providing Hashable references to arbitrary objects, along with registries of those objects'''

from typing import ClassVar, Mapping, Protocol, runtime_checkable

from uuid import UUID, uuid4
from weakref import WeakValueDictionary


@runtime_checkable
class Addressable(Protocol):
    '''Behavioral interface for types which support registered, hashable object addressing'''
    registry_addresses : ClassVar[Mapping[str, 'Addressable']]
    address : str

class Addressed: # TB DEV: should name as "AddressedMixin" explicitly?
    '''
    Mixin defining boilerplate for objects which are to be assigned a unique, hashable address at initialization
    Objects are also registered to a subclass-wide registry ("registry_addressed") keyed by their addresses
    '''
    registry_addresses : ClassVar[WeakValueDictionary[str, 'Addressed']]
    
    def __init_subclass__(cls, /,  **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry_addresses = WeakValueDictionary() # avoids sharing mutable registry with subclasses 

    # Object attr declarations
    _uuid : UUID
    _address : str

    def __new__(cls, *args, **kwargs) -> 'Addressed':
        if cls is Addressed:
            raise TypeError(f'Cannot instantiate from {cls.__name__} directly; must be used as mixin')

        obj = super(Addressed, cls).__new__(cls)

        unique_id = uuid4()
        obj._uuid = unique_id
        obj._address = unique_id.hex # opting for str conversion to avoid consumers needing to know about UUID type

        cls.registry_addresses[obj._address] = obj

        return obj

    # NOT the same as __hash__ (instances with the same hash will have different addresses)
    @property # protected, i.e. setter or deleter deliberately NOT offered
    def address(self) -> str: 
        '''Hashable hexadecimal string address unique to this object instance'''
        return self._address
    addr = address # alias for convenience