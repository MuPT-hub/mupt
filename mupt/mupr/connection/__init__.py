'''Abstractions of connections between structural units'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .types import (
    AttachmentLabel,
    ConnectorLabel,
    ConnectorHandle,
    ManagesConnectors,
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
)