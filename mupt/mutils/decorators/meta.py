'''Decorators for modifying other decorators'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Concatenate, Callable, ParamSpec, TypeAlias, TypeVar
from functools import update_wrapper, wraps

Params = ParamSpec('Params') # can also use to typehint *args and **kwargs
ReturnType = TypeVar('ReturnType') 
Decorator : TypeAlias = Callable[[Callable[Params, ReturnType]], Callable[Params, ReturnType]]


# META DECORATORS
def extend_to_methods(dec : Decorator) -> Decorator:
    '''Meta-decorator; modifies an existing decorator definition to be transferrable to methods with no additional code
    The modified decorator can be used interchangably to decorate both ordinary functions AND methods of classes'''
    ReturnSignature = dec.__annotations__.get('return')

    @wraps(dec, updated=()) # transfer decorator signature to decorator adapter class, without updating the __dict__ field
    class AdaptedDecorator:
        def __init__(self, funct : Callable[Params, ReturnType]) -> None:
            '''Record function'''
            self.funct = funct
            update_wrapper(self, funct) # equivalent to functools.wraps, transfers docstring, module, etc. for documentation

        def __call__(self, *args : Params.args, **kwargs : Params.kwargs) -> ReturnSignature: # TODO : fix this to reflect the decorator's return signature
            '''Apply decorator to function, then call decorated function'''
            return dec(self.funct)(*args, **kwargs)

        def __get__(self, instance : object, owner : type) -> Callable[[Concatenate[object, Params]], ReturnType]:
            '''Generate partial application with calling instance as first argument (fills in for "self")'''
            method = self.funct.__get__(instance, owner) # look up method belonging to owner class
            return dec(method) # return the decorated method

    return AdaptedDecorator