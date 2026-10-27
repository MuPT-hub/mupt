"""Unit tests for ConnectorManager subtypes"""

# ruff: noqa: F401

import pytest

from mupt.mupr.connection.connectors import AttachmentPoint, Connector
from mupt.mupr.connection.management import (
    ConnectorManager,
    ConnectorManagerFrozen,
    ConnectorManagerMutable,
)


def test_add_connector():
    """
    Test that Connectors can be added for mutable
    managers, and is blocked for frozen one
    """
    ...


def test_remove_connector():
    """
    Test that Connectors can be removed for mutable
    manager, and is blocked for frozen one
    """
    ...
