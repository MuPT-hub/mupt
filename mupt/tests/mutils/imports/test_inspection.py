'''Unit tests for inspecting module info'''

import pytest

from types import ModuleType
from dataclasses import dataclass

from mupt.mutils.imports.inspection import (
    _load_module,
    is_module,
    is_package,
    get_calling_module, # TODO - write test(s) for this
)

# modules not supplying any functionality, but instead used as examples in tests
import math, json # use these as test cases, since they are pretty stable in stdlib

from mupt import mupt # this is a dummy toplevel module, and NOt the entire polymerist package
from mupt.mutils import imports


# test examples
def non_module_types() ->list[type]:
    '''Types that are obviously not modules OR packages, and which should fail'''
    return [
        bool, int, float, complex, tuple, list, dict, set, 
        # str, Path # str and Path need to be tested separately
    ]

@dataclass(frozen=True)
class ModuleExample:
    '''For encapsulating package and module check tests'''
    resource : str | ModuleType
    is_module : bool
    is_package : bool

def module_examples() -> tuple[ModuleExample, ...]:
    '''
    Module-like objects labelled with whether they
    are modules and whether they are packages
    '''
    return (
        # deliberately weird to ensure this never accidentally clashes with a legit module name
        ModuleExample('--not_a_module--', False, False), 
        ModuleExample(math, True, False),
        # test that the string -> module resolver also works as intended
        ModuleExample('math', True, False), 
        ModuleExample(json, True, True),
        ModuleExample('json', True, True),
        ModuleExample(json.decoder, True, False),
        ModuleExample('json.decoder', True, False),
        ModuleExample(mupt, True, False),
        ModuleExample('mupt.mupt', True, False),
        ModuleExample(imports, True, True),
        ModuleExample('mupt.mutils.imports', True, True),
    )

# tests proper
@pytest.mark.parametrize('non_module_type', non_module_types())
def test_invalid_module_type_rejected(non_module_type : type) -> None:
    '''Check that module loading on obviously non-module types raises explicit Exception'''
    with pytest.raises(ModuleNotFoundError) as err_info:
        instance = non_module_type() # create a default instance
        _ = _load_module(instance)

@pytest.mark.parametrize('module_example', module_examples())
def test_is_module(module_example : ModuleExample) -> None:
    '''See if Python module perception behaves as expected'''
    assert is_module(module_example.resource) == module_example.is_module

@pytest.mark.parametrize('non_module_type', non_module_types())
def test_is_module_fail_on_invalid_types(non_module_type : type) -> None:
    '''Check that module perception fails on invalid input types'''
    instance = non_module_type() # create a default instance
    assert not is_module(instance)

@pytest.mark.parametrize('module_example', module_examples())
def test_is_package(module_example : ModuleExample) -> None:
    '''See if Python package perception behaves as expected'''
    assert is_package(module_example.resource) == module_example.is_package

@pytest.mark.parametrize('non_package_type', non_module_types())
def test_is_package_fail_on_invalid_types(non_package_type : type) -> None:
    '''Check that package perception fails on invalid input types'''
    instance = non_package_type() # create a default instance
    assert not is_package(instance)
