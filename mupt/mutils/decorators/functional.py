'''Decorators for modifying functions'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Concatenate, Iterable, Iterator, Optional, ParamSpec, TypeVar, Union

T = TypeVar('T')
Params = ParamSpec('Params')

from inspect import signature, Parameter
from functools import wraps, partial

from copy import deepcopy
from pathlib import Path

from .meta import extend_to_methods
from . import signatures
from ..filepaths.pathutils import aspath, asstrpath


@extend_to_methods
def optional_in_place(funct : Callable[[Concatenate[object, Params]], None]) -> Callable[[Concatenate[object, Params]], Optional[object]]:
    '''Decorator function for allowing in-place (writeable) functions which modify object attributes
    to be not performed in-place (i.e. read-only), specified by a boolean flag'''
    # TODO : add assertion that the wrapped function has at least one arg AND that the first arg is of the desired (limited) type
    old_sig = signature(funct)
    
    @wraps(funct) # for preserving docstring and type annotations / signatures
    def in_place_wrapper(obj : object, *args : Params.args, in_place : bool=False, **kwargs : Params.kwargs) -> Optional[object]: # read-only by default
        '''If not in-place, create a clone on which the method is executed''' # NOTE : old_sig.bind screws up arg passing
        if in_place:
            funct(obj, *args, **kwargs) # default call to writeable method - implicitly returns None
        else:
            copy_obj = deepcopy(obj) # clone object to avoid modifying original
            funct(copy_obj, *args, **kwargs) 
            return copy_obj # return the new object
    
    # ADD IN-PLACE PARAMETER TO FUNCTION SIGNATURE
    new_sig = signatures.insert_parameter_at_index(
        old_sig,
        new_param=Parameter(
            name='in_place',
            default=False,
            annotation=bool,
            kind=Parameter.KEYWORD_ONLY
        ),
        index=signatures.get_index_after_positionals(old_sig)
    )

    # ANNOTATE MODIFED OBJECT RETURN TYPE AS Optional[<type>]
    mod_type = tuple(old_sig.parameters.values())[0].annotation # get annotation of the object being modified
    in_place_wrapper.__signature__ = new_sig.replace(return_annotation=Optional[mod_type]) # replace signature with appropriate modifications

    return in_place_wrapper

# TODO : implement support for extend_to_methods (current mechanism is broken by additional deocrator parameters)
def flexible_listlike_input(funct : Callable[[Iterator], T]=None, CastType : type[Iterator]=list, valid_member_types : Union[type, tuple[type]]=object) -> Callable[[Iterable], T]:
    '''Wrapper which allows a function which expects a single list-initializable, Container-like object to accept any Iterable (or even star-unpacked arguments)'''
    if not issubclass(CastType, Iterator):
        raise TypeError(f'Cannot wrap listlike input with non-listlike type "{CastType.__name__}"')

    @wraps(funct)
    def wrapper(*args) -> T: # wrapper which accepts an arbitrary number of non-keyword argument
        if (len(args) == 1) and isinstance(args[0], Iterable):
            args = args[0]

        inputs = []
        for member in args:
            if isinstance(member, valid_member_types): # works because isinstance() accepts either a single type or a tuple of types
                inputs.append(member)
            else:
                raise TypeError(f'Item {member!r} of type {type(member).__name__} is not an instance of any of the following valid wrapped types: {valid_member_types}')
        inputs = CastType(inputs) # convert to the expected cast type (this is where the requirement of listlike cast types comes into play)

        return funct(inputs) # TODO: modify input type signature of wrapper function

    if funct is None:
        return partial(flexible_listlike_input, CastType=CastType, valid_member_types=valid_member_types)
    return wrapper

@extend_to_methods
def allow_string_paths(funct : Callable[[Concatenate[Path, Params]], T]) -> Callable[[Concatenate[Union[Path, str], Params]], T]:
    '''Modifies a function which expects a Path as its first argument to also accept string-paths'''
    # TODO : add assertion that the wrapped function has at least one arg AND that the first arg is of the desired (limited) type
    old_sig = signature(funct) # lookup old type signature

    @wraps(funct) # for preserving docstring and type annotations / signatures
    def str_path_wrapper(flex_path : Union[str, Path], *args : Params.args, **kwargs : Params.kwargs) -> T:
        '''First converts stringy paths into normal Paths, then executes the original function'''
        return funct(aspath(flex_path), *args, **kwargs)

    # MODIFY SIGNATURE OF PATH-LIKE FIRST ARGUMENT TO MATCH NEW TYPE FLEXIBILITY
    str_path_wrapper.__signature__ = signatures.modify_param_annotation_by_index(
        old_sig,
        index=0, # modify signature of first argument to reflect new type flexibility
        new_type=Union[Path, str]
    )

    return str_path_wrapper

@extend_to_methods
def allow_pathlib_paths(funct : Callable[[Concatenate[str, Params]], T]) -> Callable[[Concatenate[Union[Path, str], Params]], T]:
    '''Modifies a function which expects a string path as its first argument to also accept canonical pathlib Paths'''
    # TODO : add assertion that the wrapped function has at least one arg AND that the first arg is of the desired (limited) type
    old_sig = signature(funct) # lookup old type signature

    @wraps(funct) # for preserving docstring and type annotations / signatures
    def str_path_wrapper(flex_path : Union[str, Path], *args : Params.args, **kwargs : Params.kwargs) -> T:
        '''First converts normal Paths into stringy paths, then executes the original function'''
        return funct(asstrpath(flex_path), *args, **kwargs)

    # MODIFY SIGNATURE OF PATH-LIKE FIRST ARGUMENT TO MATCH NEW TYPE FLEXIBILITY
    str_path_wrapper.__signature__ = signatures.modify_param_annotation_by_index(
        old_sig,
        index=0, # modify signature of first argument to reflect new type flexibility
        new_type=Union[Path, str]
    )

    return str_path_wrapper