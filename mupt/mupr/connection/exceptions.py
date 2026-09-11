'''Exceptions specific to Connectors and related operations'''


class ConnectionError(Exception):
    '''Raised when Connector-related errors as encountered'''
    pass

class ConnectorLockedError(ConnectionError, AttributeError):
    '''Raised when attempting to modify immutable attributes on a locked Connector'''
    pass

class IncompatibleConnectorError(ConnectionError):
    '''Raised when attempting to connect two Connectors which are, for whatever reason, incompatible'''
    pass

class MissingConnectorError(ConnectionError):
    '''Raised when a required Connector is missing'''
    pass

class UnboundConnectorError(ConnectionError):
    '''Raised when a pair of Connectors are unexpectedly not bound to one another'''
    pass