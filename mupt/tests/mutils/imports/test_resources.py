'''Unit tests for fetching package resources'''

import pytest
from _pytest.mark.structures import ParameterSet

from types import ModuleType

from pathlib import Path

# modules not supplying any functionality, but instead used as examples in tests
from mupt import tests
from mupt.mutils import imports
from mupt.mutils.imports import resources
from mupt.mutils.imports.resources import (
    get_dir_path_within_package,
    get_file_path_within_package,
    get_resource_path_within_package,
)


# test examples
def obviously_fake_resource_param() -> ParameterSet:
    return pytest.param(
        'fake/whatever.txt', resources,
        marks=pytest.mark.xfail(
            raises=(TypeError, ValueError), # DEVNOTE: annoyingly, Exception raised is TypeError in Python 3.11 but ValueError in 3.12
            reason="Module is not a package and therefore cannot contain resources",
            strict=True,
        )
    )

# tests proper
@pytest.mark.parametrize(
    'rel_path, module',
    [
        ('data', tests),
        ('data/sample.dat', tests),
        pytest.param(
            'daata/simple.dat', tests,
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="This isn't a real file",
                strict=True
            ),
        ),
        ('resources.py', imports),
        obviously_fake_resource_param(),
    ]
)
def test_get_resource_path(rel_path : str, module : ModuleType) -> None:
    '''Test fetching a resource (i.e. file OR dir) from a package'''
    resource_path = get_resource_path_within_package(rel_path, module)
    assert isinstance(resource_path, Path)

@pytest.mark.parametrize(
    'rel_path, module',
    [
        pytest.param(
            'data', tests,
            marks=pytest.mark.xfail(
                raises=FileNotFoundError,
                reason="This is a directory, NOT a file",
                strict=True,
            )
        ),
        ('data/sample.dat', tests),
        pytest.param(
            'daata/simple.dat', tests,
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="This isn't a real file",
                strict=True,
            )
        ),
        ('resources.py', imports),
        obviously_fake_resource_param(),
    ]
)
def test_get_file_path(rel_path : str, module : ModuleType) -> None:
    '''Test fetching a file (i.e. NOT a dir) from a package'''
    resource_path = get_file_path_within_package(rel_path, module)
    assert isinstance(resource_path, Path)

@pytest.mark.parametrize(
    'rel_path, module',
    [
        ('data', tests),
        pytest.param(
            'data/sample.dat', tests,
            marks=pytest.mark.xfail(
                raises=NotADirectoryError,
                reason='This IS a real file, but not a directory',
                strict=True,
            )
        ),
        pytest.param(
            'daata/simple.dat', tests,
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="This isn't a real file",
                strict=True,
            )
        ),
        pytest.param(
            'resources.py', imports, 
            marks=pytest.mark.xfail(
                raises=NotADirectoryError,
                reason='This IS a real file, but not a directory',
                strict=True,
            )
        ),
        obviously_fake_resource_param(),
    ]
)
def test_get_dir_path(rel_path : str, module : ModuleType) -> None:
    '''Test fetching a dir (i.e. NOT a file) from a package'''
    resource_path = get_dir_path_within_package(rel_path, module)
    assert isinstance(resource_path, Path)