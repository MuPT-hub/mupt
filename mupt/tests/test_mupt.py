"""
Unit and regression test for the mupt package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mupt


def test_mupt_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mupt" in sys.modules
