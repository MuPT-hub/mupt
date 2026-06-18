'''Managed collection of Connectors - used to outsource business logic from Primitive'''

from typing import (
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
)
from abc import abstractmethod

from .connectors import Connector
from .types import (
    ConnectorLabel,
    ConnectorHandle,
    ConnectorAddress,
)
from mupt.mutils.containers import (
    UniqueRegistry,
    Labelled,
    LabelledT,
    HandleT,
) 


class HoldsConnectors(Protocol):
    '''
    Type indicator for another class which is in some sense a 'proprietor' of
    a collection of Connectors, but employs a ConnectorManager to manage them
    '''
    connections : ConnectorManager

class ConnectorManager(Protocol):
    '''Interface for generic connector managment object'''
    connectors : Collection['Connector']
    connectors_by_handle : Mapping[ConnectorAddress, 'Connector']

    @abstractmethod
    def connector(self, conn_addr : ConnectorAddress) -> 'Connector':
        ...

    @property
    @abstractmethod
    def connectors_free(self) -> Collection['Connector']:
        '''Connectors which are currently unbound'''
        ...

    @property
    @abstractmethod
    def connectors_bound(self) -> Collection['Connector']:
        '''Connectors which have a neighbor'''
        ...

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

class ConnectorManagerFrozen(ConnectorManager):
    '''
    ConnectorManager which does not permit mutation to connectivity after creation
    '''
    _connectors_all : tuple[Connector, ...]
    _connectors_free : tuple[Connector, ...]
    _connectors_bound : tuple[Connector, ...]

    def __new__(
        cls,
        connectors : Iterable[Connector],
        connectors_free  : Optional[Iterable[Connector]]=None,
        connectors_bound : Optional[Iterable[Connector]]=None,
    ) -> object:
         # TODO: make Registries and set labels procedurally (somehow)
        obj = super(ConnectorManagerFrozen, cls).__new__(cls)
        obj._connectors_all = tuple(connectors)

        if connectors_free is None:
            connectors_free = tuple(
                conn
                    for conn in connectors
                        if conn.neighbor is None
            )
        obj._connectors_free = tuple(connectors_free) # will take caller's word for it

        if connectors_bound is None:
            connectors_bound = tuple(
                conn
                    for conn in connectors
                        if conn.neighbor is not None
            )
        obj._connectors_bound = tuple(connectors_bound) # will take caller's word for it

        return obj

    def connector(self, conn_addr : ConnectorAddress) -> Connector:
        return self._connectors_all[conn_addr]

    @property
    def connectors(self) -> tuple[Connector, ...]:
        return self._connectors_all

    @property
    def connectors_free(self) -> Collection[Connector]:
        '''
        Connectors whose have not yet been assigned a neighbor
        '''
        return self._connectors_free
        
    @property
    def connectors_bound(self) -> Collection[Connector]:
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
        ...

    def connector(self, conn_addr : ConnectorAddress) -> 'Connector':
        ...

    @property
    def connectors_free(self) -> Collection['Connector']:
        '''Connectors which are currently unbound'''
        ...

    @property
    def connectors_bound(self) -> Collection['Connector']:
        '''Connectors which have a neighbor'''
        ...