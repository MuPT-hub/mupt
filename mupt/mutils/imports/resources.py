'''For loading non-module data shipped with a Python library'''

from typing import Union
from pathlib import Path

from importlib.resources import files, Package
from importlib.resources.abc import Traversable
from importlib.resources._common import resolve


def get_resource_path_within_package(relative_path : Union[str, Path], package : Package) -> Path:
    '''Get the Path to a resource (i.e. either a directory or a file) which lives within a Python package'''
    package_path : Traversable = files(package) # also implicitly checks the provided package exists
    if not isinstance(package_path, Path):
        raise TypeError(f'Expected path to package "{package}" to be returned as Pathlike, got {type(package_path).__name__} instead')

    resource_path = package_path / relative_path    # concat to Path here means string inputs for relative_path are valid without explicit conversion
    if not resource_path.exists(): # if this block is reached, it means "package" is a real module and resource path is DEFINED relative to package's path, so the below message is valid
        raise ValueError(f'{resolve(package).__name__} contains no resource "{relative_path}"')
    
    return resource_path

def get_dir_path_within_package(relative_path : Union[str, Path], package : Package) -> Path:
    '''Get the Path to a directory which lives within a Python package'''
    dir_path : Path = get_resource_path_within_package(package=package, relative_path=relative_path)
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f'{resolve(package).__name__} contains "{dir_path}", but it is not a directory')
    
    return dir_path

def get_file_path_within_package(relative_path : Union[str, Path], package : Package) -> Path:
    '''Get the Path to a (non-directory) file which lives within a Python package'''
    file_path : Path = get_resource_path_within_package(package=package, relative_path=relative_path) 
    
    if not file_path.is_file():
        raise FileNotFoundError(f'{resolve(package).__name__} contains no file "{file_path}"')
    
    return file_path