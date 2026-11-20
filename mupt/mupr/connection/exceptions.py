"""Exceptions specific to Connectors and related operations"""


class ConnectionError(Exception):
    """Raised when Connector-related errors as encountered"""

    pass


class IncompatibleConnectorError(ConnectionError):
    """
    Raised when attempting to connect two Connectors
    which are, for whatever reason, incompatible
    """

    pass


class MissingConnectorError(ConnectionError):
    """Raised when a required Connector is missing"""

    pass


class UnboundConnectorError(ConnectionError):
    """
    Raised when a pair of Connectors are
    unexpectedly not bound to one another
    """

    pass


class ConnectorAntialignmentError(ConnectionError, ValueError):
    """
    Raised when a pair of Connectors which are
    expected to be spatially antialigned are not
    """

    pass
