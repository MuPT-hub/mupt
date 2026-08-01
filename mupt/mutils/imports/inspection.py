'''
For checking whether object are valid Python modules and packages
and if so for gathering information about them and  from within them
'''

from typing import Optional, Union
from types import ModuleType

from inspect import stack, getmodule
from importlib.resources import Package
from importlib.resources._common import resolve

ModuleLike = Union[str, ModuleType, Package]


def get_calling_module(stacklevel : int=1) -> Optional[ModuleType]:
    '''When invoked within a Callable, return the module from which that Callable was called'''
    try:
        caller_frame_info = stack()[stacklevel]
    except IndexError:
        return None
    return getmodule(caller_frame_info.frame)

def _load_module(module : ModuleLike) -> ModuleType:
    '''
    Type-safe flexible resource load - raises ModuleNotFoundError if no ModuleType can be loaded
    (to restrict the relatively-permissive importlib.resources._common.resolve())
    '''
    try:
        module_loaded = resolve(module) # if string-y, will raise ModuleNotFoundError by default if the module doesn't exist
    except AttributeError:
        raise ModuleNotFoundError # Coercion needed to raise uniform Exception in testing between 3.11 and 3.12
    # TODO: add mechanism for loading from Path

    if not hasattr(module_loaded, '__spec__'):
        raise ModuleNotFoundError
    
    return module_loaded

def is_module(module : ModuleLike) -> bool:
    '''Determine whether a given Package-like (i.e. str or ModuleType) is a valid Python module
    This will return True for packages, bottom-level modules (i.e. *.py) and Python scripts'''
    try:
        _ = _load_module(module)
        return True # enough to check that module is loadable as ModuleType
    except ModuleNotFoundError:
        return False
    
def is_package(package : ModuleLike) -> bool:
    '''Determine whether a given Package-like (i.e. str or ModuleType) is a valid Python package'''
    try:
        # per importlib docs (https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.submodule_search_locations):
        # "[submodule_search_locations] should be set to None for non-package modules"
        module_loaded : ModuleType = _load_module(package)
        return (module_loaded.__spec__.submodule_search_locations is not None)
    except ModuleNotFoundError:
        return False