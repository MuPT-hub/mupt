"""
Utilities for providing Hashable references to 
arbitrary objects, along with registries of those objects
"""

from typing import ClassVar, Mapping, Protocol, runtime_checkable

from uuid import UUID, uuid4
from weakref import WeakValueDictionary


@runtime_checkable
class Addressable(Protocol):
    """
    Behavioral interface for types which support
    registered, hashable object addressing
    """

    registry_addresses: ClassVar[Mapping[str, "Addressable"]]
    address: str

class Addressed:  # TB DEV: should name as "AddressedMixin" explicitly?
    """
    Mixin defining boilerplate for objects which are to be 
    assigned a unique, hashable address during construction.
    
    Objects are also registered to a subclass-wide registry
    (attr named `registry_addresses`) keyed by their addresses.
    """

    registry_addresses: ClassVar[WeakValueDictionary[str, "Addressed"]]

    def __init_subclass__(cls, /, **kwargs) -> None:
        """
        Initialize subclass-specific address-to-object registry
        
        Done to avoid cross-contamination of instances between disparate classes
        Addressed is intended to behave as a mixin with no other shared behaviors
        """
        super().__init_subclass__(**kwargs)
        cls.registry_addresses = (
            WeakValueDictionary()
        )  # avoids sharing mutable registry with subclasses

    # Object attr declarations
    _uuid: UUID
    _address: str

    def __new__(cls, *args, **kwargs) -> "Addressed":
        """
        Create new instance, assign it a unique address, and register
        the address : instance key-value pair in the class' internal registry
        
        Retuirn the created instance
        """
        if cls is Addressed:
            raise TypeError(
                f"Can't instantiate from {cls.__name__} directly; must be used as mixin"
            )

        obj = super(Addressed, cls).__new__(cls)

        unique_id = uuid4()
        obj._uuid = unique_id
        # opting for str conversion to avoid consumers needing to know about UUID type
        obj._address = unique_id.hex

        cls.registry_addresses[obj._address] = obj

        return obj

    # NOT the same as __hash__ (instances w/ same hash will have different addresses)
    @property  # protected, i.e. setter or deleter deliberately NOT offered
    def address(self) -> str:
        """Hashable hexadecimal string address unique to this object instance"""
        return self._address

    addr = address  # alias for convenience
