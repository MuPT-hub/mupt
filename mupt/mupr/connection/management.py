'''Managed collection of Connectors - used to outsource business logic from Primitive'''

from typing import (
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
)
from types import MappingProxyType

from .connectors import Connector
from .types import (
    ConnectorAddress,
    ConnectorLabel,
    ConnectorHandle,
    ConnectorLabeller,
)
from ...mutils.containers import (
    UniqueRegistry,
    Labelled,
    LabelledT,
    HandleT,
) 


class ConnectorManager(Protocol):
    '''Interface for generic connector managment object'''
    connectors : Collection[Connector]
    connectors_free : Collection[Connector]
    connectors_bound : Collection[Connector]
    connectors_by_addr : Mapping[ConnectorAddress, Connector]
    connectors_by_handle : Mapping[ConnectorHandle, Connector]

    def connector(self, conn_addr : ConnectorAddress) -> Connector:
        '''Retrieve a particular Connector by its unique address'''
        return self.connectors_by_addr[conn_addr] # not using .get() to make KeyErrors explicit

    # default implementations, for when explicitly inherited
    @property
    def functionality(self) -> int:
        return len(self.connectors_free)
    
    @property
    def valence(self) -> int: # DEV: well-defined from more than just atomic primitives since Connectors store BondType info
        '''Electronic valence of the Primitive, i.e. the total bond order of all external-facing Connectors on this Primitive'''
        total_bond_order : float = sum(conn.bond_order for conn in self.connectors)
        return round(total_bond_order)
    chemical_valence = electronic_valence = valence # aliases for convenience

class HoldsConnectors(Protocol):
    '''
    Type indicator for another class which is in some sense a 'proprietor' of
    a collection of Connectors, but employs a ConnectorManager to manage them
    '''
    connections : ConnectorManager

# Concrete ConnectorManager types
class ConnectorManagerFrozen(ConnectorManager):
    '''
    ConnectorManager which does not permit mutation to connectivity after creation
    '''
    _connectors_all : tuple[Connector, ...]
    _connectors_free : tuple[Connector, ...]
    _connectors_bound : tuple[Connector, ...]
    _connectors_by_addr : MappingProxyType[ConnectorAddress, Connector]

    def __new__(
        cls,
        connectors : Iterable[Connector],
        # TODO: provide optimization short-circuit to allow making use of known free/bound designations
        connectors_free  : Optional[Iterable[Connector]]=None,
        connectors_bound : Optional[Iterable[Connector]]=None,
    ) -> object:
         # TODO: make Registries and set labels procedurally (somehow)
        obj = super(ConnectorManagerFrozen, cls).__new__(cls)
        obj._connectors_all = tuple(connectors)
        obj._connectors_by_addr = MappingProxyType({conn.address : conn for conn in connectors})

        connectors_free_accum : list[Connector] = [] 
        connectors_bound_accum : list[Connector] = [] 
        for conn in connectors:
            # TB DEV: lock here is not secure as yet, since one could manually unlock after init
            conn.lock() # ensure not mutations allowed subsequently
            if conn.has_neighbor:
                connectors_bound_accum.append(conn)
            else:
                connectors_free_accum.append(conn)
        obj._connectors_free  = tuple(connectors_free_accum)
        obj._connectors_bound = tuple(connectors_bound_accum)

        return obj
    
    @property
    def connectors_by_addr(self) -> Mapping[ConnectorAddress, Connector]:
        return self._connectors_by_addr

    @property
    def connectors(self) -> tuple[Connector, ...]:
        return self._connectors_all

    @property
    def connectors_free(self) -> tuple[Connector, ...]:
        '''
        Connectors whose have not yet been assigned a neighbor
        '''
        return self._connectors_free
        
    @property
    def connectors_bound(self) -> tuple[Connector, ...]:
        '''
        Connectors (originating from children as they must) which are
        bound and whose neighbor is also a child of this Composite
        '''
        return self._connectors_bound

class ConnectorManagerMutable(ConnectorManager):
    '''
    ConnectorManager with mutable connections
    Necessary for configuring initial connectivity
    '''
    def __init__(
        self,
        connectors : Iterable[Connector],
        default_label : Hashable='CONN',
    ) -> None:
        self.connectors_by_addr : dict[ConnectorAddress, Connector] = {}
        for conn in connectors:
            conn.unlock()
            self.add_connector(conn)

    def add_connector(
        self,
        conn : Connector,
        label : Optional[ConnectorLabel | ConnectorLabeller]=None,
    ) -> None:
        '''Register a new Connector to be managed here'''
        # TODO: label to be used for UniqueRegistry registration to give human-readable handle
        self.connectors_by_addr[conn.addr] = conn

    def remove_connector(
        self,
        conn_addr : ConnectorAddress | Connector,
    ) -> Connector:
        '''Declare a Connector to be no longer managed here'''
        if isinstance(conn_addr, Connector):
            conn_addr = conn_addr.address
        
        return self.connectors_by_addr.pop(conn_addr)

    @property
    def connectors(self) -> tuple[Connector, ...]:
        return tuple(self.connectors_by_addr.values())
    
    # DEV: opting for linear search each time (rather than dynamically-updating list)
    # since connectors might change neighbor status during bond linking (checks when called)
    @property
    def connectors_free(self) -> tuple[Connector, ...]:
        '''Managed Connectors which have no assigned neighbor'''
        return tuple(conn for conn in self.connectors if not conn.has_neighbor)

    @property
    def connectors_bound(self) -> tuple[Connector, ...]:
        '''Managed Connectors which have no assigned neighbor'''
        return tuple(conn for conn in self.connectors if conn.has_neighbor)