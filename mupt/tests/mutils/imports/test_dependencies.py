'''Unit tests for `dependencies` package'''

import pytest

from typing import Union
from mupt.mutils.imports.dependencies import (
    modules_installed,
    requires_modules,
    MissingPrerequisitePackage,
)


def test_missing_prereq_call_module() -> None:
    '''Test that auto-deduced calling module is accurate'''
    with pytest.raises(MissingPrerequisitePackage, match=__name__): # check *THIS* module appears in the error msg
        raise MissingPrerequisitePackage('bogus-module', importing_package_name=None)

@pytest.mark.parametrize(
    'module_names, expected_found', [
        (['mupt'], True),     # we'd better hope the parent module is present if we're running tests on it :P
        (['sys'], True),            # test stdlib packages which ought to be present if Python is
        (['os', 'sys'], True),      # test that unpacking also works
        (['fake--module'], False),  # test an obviously fake module name (don't want to try an actual module in case it becomes an dependency someday)
        ([42], False),              # test something that isn't even a module to check error handling
    ]
)
def test_modules_installed(module_names : list[str], expected_found : bool) -> None:
    '''Check that module install checker correctly identifies present and absent modules'''
    # TB: also implicitly tests module_installed (singular); worth having explicit tests for that?
    assert modules_installed(*module_names) == expected_found
    
# Testing requires_modules() decorator
@pytest.mark.parametrize(
    'module_name,missing_module_error',
    [
        ('os', ImportError),
        ('os', ImportError('This is not the default message!')),
        pytest.param(
            'os', 42, # note that module IS valid here but the Exception raised is not (meta-error)
            marks=pytest.mark.xfail(
                raises=TypeError, 
                reason='Non Exception-like object passed to missing_module_error',
                strict=True,
            )
        ),
        # Test that multiple injected Exception types are supported
        pytest.param(
            'fake--module', ImportError, 
            marks=pytest.mark.xfail(
                raises=ImportError, # NOTE that this ImportError MUST be the same as the import error passed via args
                reason='The required module shouldn\'t be found in the environment',
                strict=True,
            )
        ),
        pytest.param(
            'fake--module', AttributeError('something else'), 
            marks=pytest.mark.xfail(
                raises=AttributeError, # NOTE that this AttributeError MUST be the same as the import error passed via args
                reason='The required module shouldn\'t be found in the environment',
                strict=True,
            )
        ),
    ]
)
def test_requires_modules(module_name : str, missing_module_error : Union[Exception, type[Exception]]) -> None:
    '''Test that the requires_modules decorator correctly wraps functions'''
    @requires_modules(module_name, missing_module_error=missing_module_error)
    def func() -> str:
        return 'I am pointless'
    
    _ = func() # no assertion needed, xfail cases should raise Exception while working cases will terminate without Exception