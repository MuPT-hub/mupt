'''Typehints, alises, and Protocols relating to how connections are specified'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Collection,
    Hashable,
    Mapping,
    Protocol,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from .connectors import Connector

type AttachmentLabel = Hashable  # TODO: narrow down this type as use cases become clearer
type ConnectorLabel = Hashable
ConnectorHandle = tuple[ConnectorLabel, int]

type ConnectorAddress = int # DEV TB: consider if this type needs to be more specific than just an alias


class ManagesConnectors(Protocol):
    '''Interface for objects which manage Connectors and pairs of Connectors ("connections")'''
    connectors : Collection['Connector']
    connectors_by_address : Mapping[ConnectorAddress, 'Connector']
    
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