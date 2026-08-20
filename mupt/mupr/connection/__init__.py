'''Abstractions of connections between structural units'''

from .types import (
    AttachmentLabel,
    ConnectorLabel,
    ConnectorHandle,
)
from .exceptions import (
    ConnectionError,
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
from .connectors import (
    AttachmentPoint,
    Connector,
    canonical_form_connectors,
)