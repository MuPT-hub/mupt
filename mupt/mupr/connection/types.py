'''Typehints, aliases, and Protocols relating to how connections are specified'''

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
type ConnectorHandle = tuple[ConnectorLabel, int]
type ConnectorAddress = int # DEV TB: consider if this type needs to be more specific than just an alias
