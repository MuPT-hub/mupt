"""Typehints, aliases, and Protocols relating to how connections are specified"""

from typing import (
    Callable,
    Hashable,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .connectors import Connector


type AttachmentLabel = Hashable

# TB: consider if this type needs to be more specific
type ConnectorAddress = Hashable
type ConnectorLabel = Hashable
type ConnectorLabeller = Callable[[Connector], ConnectorLabel]
type ConnectorLabelLike = Union[ConnectorLabel, ConnectorLabeller]
type ConnectorHandle = tuple[ConnectorLabel, int]
