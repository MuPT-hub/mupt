'''Utilities for providing Hashable references to arbitrary objects, along with registries of those objects'''

from typing import ClassVar
from uuid import UUID, uuid4
from weakref import WeakValueDictionary


class Addressable:
    '''Boilerplate for objects which are assigned a universally-unique ID at initialization'''
    registry_addresses : ClassVar[WeakValueDictionary[str, 'Addressable']]
    
    def __init_subclass__(cls, /,  **kwargs) -> None:
        super(cls).__init_subclass__(**kwargs)
        cls.registry_addresses = WeakValueDictionary() # avoids sharing mutable registry with subclasses 

    # Object attr declarations
    _uuid : UUID
    _address : str

    # def __init__(self) -> None:
    def __new__(cls, *args, **kwargs) -> 'Addressable':
        obj = super(Addressable, cls).__new__(cls)

        unique_id = uuid4()
        obj._uuid = unique_id
        obj._address = unique_id.hex # opting for str conversion to avoid consumers needing to know about UUID type

        cls.registry_addresses[obj._address] = obj

        return obj

    # NOT the same as __hash__ (instances with the same hash will have different addresses)
    @property # protected, i.e. setter or deleter deliberately NOT offered
    def address(self) -> str: 
        '''Hashable hexademical string address unique to this object instance'''
        return self._address
    addr = address # alias for convenience