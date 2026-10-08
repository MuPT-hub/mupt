"""Abstractions of connections between structural units"""

from .types import (
    AttachmentLabel as AttachmentLabel,
    ConnectorLabel as ConnectorLabel,
    ConnectorLabeller as ConnectorLabeller,
    ConnectorLabelLike as ConnectorLabelLike,
    ConnectorHandle as ConnectorHandle,
)
from .exceptions import (
    ConnectionError as ConnectionError,
    IncompatibleConnectorError as IncompatibleConnectorError,
    MissingConnectorError as MissingConnectorError,
    UnboundConnectorError as UnboundConnectorError,
)
from .connection import (
    AttachmentPoint as AttachmentPoint,
    Connector as Connector,
    ConnectorSelector as ConnectorSelector,
    make_second_resemble_first as make_second_resemble_first,
)
