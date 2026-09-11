'''Typehints, aliases, and Protocols relating to how connections are specified'''

from typing import (
    Callable,
    Hashable,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from .connectors import Connector

type AttachmentLabel = Hashable  # TODO: narrow down this type as use cases become clearer
type ConnectorAddress = Hashable # DEV TB: consider if this type needs to be more specific
type ConnectorLabel = Hashable
type ConnectorLabeller = Callable[[Connector], ConnectorLabel]
type ConnectorHandle = tuple[ConnectorLabel, int] # TB: usused but leaving in case we want something like this later
