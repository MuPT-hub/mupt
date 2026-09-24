"""Abstractions of connections between structural units"""

from .types import (
    AttachmentLabel as AttachmentLabel,
    ConnectorAddress as ConnectorAddress,
    ConnectorLabel as ConnectorLabel,
    ConnectorLabeller as ConnectorLabeller,
    ConnectorHandle as ConnectorHandle,
)
from .exceptions import (
    ConnectionError as ConnectionError,
    IncompatibleConnectorError as IncompatibleConnectorError,
    MissingConnectorError as MissingConnectorError,
    UnboundConnectorError as UnboundConnectorError,
)
from .connectors import (
    AttachmentPoint as AttachmentPoint,
    Connector as Connector,
    canonical_form_connectors as canonical_form_connectors,
)
