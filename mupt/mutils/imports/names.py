'''For generating concise and convenient labels for ModuleLike objects'''

from typing import Optional
from types import ModuleType

from importlib.resources._common import resolve


def resolve_module_name(module : Optional[ModuleType | str]) -> str:
    '''Given a ModuleType, returns a str-type name for it based on its ModuleSpec (or lack thereof)'''
    if isinstance(module, str):
        module = resolve(module)

    if module is None:
        importing_package_name : str = 'unknown'
    elif (calling_module_spec := module.__spec__) is None:
        # see https://docs.python.org/3/reference/import.html#main-spec
        importing_package_name : str = '__main__'
    else:
        importing_package_name = calling_module_spec.name

    return importing_package_name

def module_parts(module : ModuleType | str) -> tuple[Optional[str], str]:
    '''Takes a module (as its name or as ModuleType) and returns its parent package name and relative module name'''
    module_name : str = resolve_module_name(module)
    parent_package_name, _, module_stem = module_name.rpartition('.') # split on rightmost dot separator
    if not parent_package_name:
        parent_package_name = None

    return parent_package_name, module_stem

def module_stem(module : ModuleType | str) -> str:
    '''Takes a module (as its name or as ModuleType) and returns its relative module name'''
    return module_parts(module)[-1]

def relative_module_name(
    module : ModuleType | str,
    relative_to : Optional[ModuleType | str]=None,
    remove_leading_dot : bool=True,
) -> str:
    '''Gets the name of a module relative to another (presumably toplevel) module
    If the given module is not in the path of the toplevel module, will simply return as module.__name__'''
    rel_mod_name = resolve_module_name(module)
    if relative_to is not None:
        toplevel_prefix = resolve_module_name(relative_to)
        if remove_leading_dot:
            toplevel_prefix += '.' # append dot to prefix to remove it later
        rel_mod_name = rel_mod_name.removeprefix(toplevel_prefix)

    return rel_mod_name
