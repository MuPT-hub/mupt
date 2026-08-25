'''Abstractions of connections between structural units'''

from .types import (
    AttachmentLabel,
    ConnectorLabel,
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