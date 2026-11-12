"""Tests for exporting MuPT hierarchies to GMSO Topologies"""

import pytest

pytest.importorskip("gmso")

from mupt.interfaces.mosdef.gmso import exporters
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.mupr.primitives.predicates import is_atom


def test_gmso_exporter_imports() -> None:
    """The GMSO exporter module imports (it wraps the mBuild exporter)"""
    assert callable(exporters.to_gmso)


def test_to_gmso_atom_sites_and_bonds() -> None:
    """Exporting at the atom level gives one GMSO site per atom and every bond"""
    ethanol = primitive_from_smiles("CCO", embed_positions=True)
    topology = exporters.to_gmso(ethanol, predicate=is_atom, coords_to_nm=0.1)
    assert topology.n_sites == 9
    assert topology.n_bonds == 8


def test_to_gmso_requires_predicate() -> None:
    """Exporting without a predicate is not supported yet"""
    ethanol = primitive_from_smiles("CCO", embed_positions=True)
    with pytest.raises(NotImplementedError):
        exporters.to_gmso(ethanol)
