"""Unit tests for Connector bond antialignment protocols"""

# ruff: noqa: F401

import pytest

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.mupr.connection.connectors import Connector, AttachmentPoint
from mupt.mupr.connection.alignment import (
    ConnectorAntialignmentStrategy,
    ConnectorAntialignmentRigid,
    ConnectorAntialignmentBallistic,
)
