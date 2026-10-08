"""Abstractions of connections between structural units"""

from .connection import (
    AttachmentPoint as AttachmentPoint,
    Connector as Connector,
    ConnectorLabel as ConnectorLabel,
    ConnectorHandle as ConnectorHandle,
    ConnectorSelector as ConnectorSelector,
    make_second_resemble_first as make_second_resemble_first,
    IncompatibleConnectorError as IncompatibleConnectorError,
    MissingConnectorError as MissingConnectorError,
    UnboundConnectorError as UnboundConnectorError,
)
