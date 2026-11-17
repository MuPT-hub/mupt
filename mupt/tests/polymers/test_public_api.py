"""The public API: every subpackage's names are re-exported from mupt.polymers"""

import pytest

import mupt.polymers
from mupt.polymers import generators, models, rules, utils


def public_names(module) -> set[str]:
    return {name for name in vars(module) if not name.startswith("_")} - {
        "annotations",
    }


@pytest.mark.parametrize("subpackage", [models, rules, utils, generators])
def test_subpackage_names_are_reexported(subpackage) -> None:
    submodules = {name for name, value in vars(subpackage).items() if type(value) is type(mupt)}
    for name in public_names(subpackage) - submodules:
        assert getattr(mupt.polymers, name) is getattr(subpackage, name), name


def test_the_main_entry_points_are_top_level() -> None:
    from mupt.polymers import (  # noqa: F401
        BlockCopolymerGenerator,
        CompatibleSites,
        LadderPolymerGenerator,
        LinearPolymerGenerator,
        RandomFlip,
        SizeSchulzZimm,
        StarPolymerGenerator,
        universe_from_descriptive_graphs,
    )
