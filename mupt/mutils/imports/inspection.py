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


def get_calling_module(levels_above : int=0) -> Optional[ModuleType]:
    '''
    Return the module inside of which *THIS* function is called
    
    Parameters
    ----------
    levels_above : int, default 0
        The number of levels up on the call stack to reference
        relative to where this function was called

        As an example, given the following 2 files:
        In foo.py:
        | from mupt.mutils.imports.inspection import get_calling_moudle
        | 
        | def show():
        |     print(get_calling_module(levels_above=?).__name__)

        In bar.py:
        | from foo import show()
        | show()

        would print 'foo' if ? is 0, and 'bar' if ? is 1
    '''
    try:
        stacklevel : int = levels_above + 1 # if 0, would always just return *THIS* module
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
        # if string-y, will raise ModuleNotFoundError by default if the module doesn't exist
        module_loaded = resolve(module) 
    except AttributeError:
        # Coercion needed to raise uniform Exception in testing between 3.11 and 3.12
        raise ModuleNotFoundError 
    # TODO: add mechanism for loading from Path

    if not hasattr(module_loaded, '__spec__'): #TBDEV: exception is the module is __main__ in certain cases; worth addressing?
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