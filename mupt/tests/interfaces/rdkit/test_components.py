"""Unit tests for RDKit component initializers (Connector labelling)"""

import pytest

from mupt.builders.heading import TraversalDirection
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.interfaces.rdkit.components import DEFAULT_CONNECTOR_LABELLER
from mupt.mupr.connection import AttachmentPoint, Connector


def free_labels(prim) -> list:
    return sorted((conn.label for conn in prim.connections.connectors_free), key=str)


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("O[*:3]", [3]),                         # linker map number
        ("[O:3]*", [3]),                         # anchor map number fallback
        ("[*:5][O:3]", [5]),                     # linker takes precedence over anchor
        ("C*", [Connector.DEFAULT_LABEL]),       # unmapped falls back to class default
        ("[*:1][CH2][*:2]", [1, 2]),             # multiple ports on one atom stay distinct
        ("*-[CH2:1]-[CH2:2]-*", [1, 2]),         # legacy 1/2 head-tail convention
        ("[*:7][CH2:1][CH:2](C)*", [2, 7]),      # mixed linker- and anchor-labelled ports
    ],
)
def test_default_labeller_labels_ports_by_map_number(smiles: str, expected: list) -> None:
    assert free_labels(primitive_from_smiles(smiles)) == expected


def test_default_labeller_labels_survive_copy() -> None:
    prim = primitive_from_smiles("[*:7][CH2:1][CH:2](C)*")
    assert free_labels(prim.copy()) == free_labels(prim)


def test_default_labeller_does_not_enforce_unique_labels() -> None:
    # documents current behavior; uniqueness is validated by consumers (e.g. topology generators)
    assert free_labels(primitive_from_smiles("[*:1]C[*:1]")) == [1, 1]


def test_imported_labels_are_not_callables() -> None:
    # regression: importer previously stored the labeller function itself as the label
    prim = primitive_from_smiles("CC*")
    assert not any(callable(conn.label) for atom in prim.children for conn in atom.connections.connectors)


def test_custom_connector_labeller_is_applied() -> None:
    prim = primitive_from_smiles("O[*:3]", connector_labeller=lambda conn: "X")
    assert free_labels(prim) == ["X"]


@pytest.mark.parametrize(
    "attachables,expected",
    [
        ({1, TraversalDirection.ANTERO}, 1),     # direction tags are ignored, not compared
        ({"c"}, Connector.DEFAULT_LABEL),        # non-map-number attachables are ignored
    ],
)
def test_default_labeller_considers_only_integer_map_numbers(attachables: set, expected) -> None:
    connector = Connector(linker=AttachmentPoint(attachables))
    assert DEFAULT_CONNECTOR_LABELLER(connector) == expected