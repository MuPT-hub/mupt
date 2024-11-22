'''Decorators for modifying classes'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Iterable, Optional, TypeVar, Union
C = TypeVar('C')


def generate_repr(cls : Optional[C]=None, disp_attrs : Optional[Iterable[str]]=None, lookup_attr : Optional[str]=None) -> Union[C, Callable[[C], C]]:
    '''
    Class decorator for auto-generating __repr__ methods

    By default (i.e. with no arguments), generated repr simply returns the name of the class
    If collection of "disp_attrs" is provided, will display the values of <disp_attrs> for the object instance being represented in series
    If "disp_attrs" is NOT provided but "lookup_attr" is, disp_attrs will be looked up from the modified parent class
    '''
    if disp_attrs is None: 
        disp_attrs = [] # set to empty list to avoid mutable default
    
    def class_decorator(cls : C) -> C:
        '''The actual (argument-free) class decorator'''
        nonlocal disp_attrs # avoids multiple scope issues in disparate use cases (refers to the variable in the outermost scope)
        if not disp_attrs and lookup_attr: # only use lookup if one is explicitly provided and no display attributes are provided
            assert(hasattr(cls, lookup_attr))
            disp_attrs = getattr(cls, lookup_attr) # if a lookup attribute is provided, lookup the attribute names within the class being modified

        def _repr_generic(self) -> str:
            attr_str = ', '.join(f'{attr}={getattr(self, attr)}' for attr in disp_attrs)
            return f'{cls.__name__}({attr_str})'
        setattr(cls, '__repr__', _repr_generic)

        return cls
    
    if cls is None: # null case (i.e. call without parens), return factory call
        return class_decorator
    return class_decorator(cls) # return literal class decorator call
        
def register_subclasses(cls : Optional[C]=None, key_attr : str='__name__', reg_attr : str='subclass_registry') -> Union[C, Callable[[C], C]]:
    '''
    Parametric class decorator for automatically generating a registry of subclasses of a target class
    Binds registry to the "registry" class property in the target class
    
    Subclasses are keyed by lookup of a target attribute <key_attr> in the child classes (by default just the name of the subclass),
    while the resulting registry class property is bound to the <reg_attr> attribute of the parent class
    '''
    def class_decorator(cls : C) -> C:
        '''The actual (argument-free) class decorator'''
        @classmethod # method should be accessible class-wide
        @property    # make property to allow for dynamic subclassing (generated at runtime, not compile time)
        def _registry(cls : C) -> dict[str, C]:
            return { # Keep a registry of all charger implementations for convenience
                getattr(subclass, key_attr) : subclass
                    for subclass in cls.__subclasses__()
            }
        setattr(cls, reg_attr, _registry) # bind registry class property to target class. TODO : check for registry already present in class
        
        return cls # return back the modified class
    
    if cls is None: # null case (i.e. call without parens), return factory call
        return class_decorator
    return class_decorator(cls) # return literal class decorator call

# NOTE: "klass" is needed to distinguish between the class modified by this decorator and the classmethod arg when calling super()
# "klass" here is the parent, while "cls" is the child
def register_abstract_class_attrs(*attr_names : list[str]) -> Callable[[C], C]: # TODO: add mechanism for typehinting
    '''Register a list of string attribute names as abstract class attributes, 
    which MUST be implemented by child classes of the wrapped class'''
    def class_decorator(klass : C) -> C:
        '''The actual (argument-free) class decorator'''
        def __wrapped_init_subclass__(cls : C, **kwargs) -> None:
            '''Wrapper for subclass definition which actually enforces that all named attributes are set'''
            for attr_name in attr_names:
                passed_attr_value = kwargs.pop(attr_name, NotImplemented) # want this removed from kwargs before passing to super, regardless of whether already set in child 
                attr_val_on_child = getattr(cls, attr_name, NotImplemented) # check if this has been set in the child in code
                
                if attr_val_on_child is NotImplemented:         # if the value has not been set in code...
                    if passed_attr_value is not NotImplemented: # ...fall back to value passed into class definition, if it exists...
                        setattr(cls, attr_name, passed_attr_value)
                    else:                                       # otherwise, fail and raise Exception
                        raise TypeError(f"Can't instantiate abstract class {cls.__name__} with abstract class property '{attr_name}' undefined")

            super(klass, cls).__init_subclass__(**kwargs) # this should fail if extraneous named args are passed

        klass.__init_subclass__ = classmethod(__wrapped_init_subclass__)
        return klass

    return class_decorator # no need for application check here, since the parameterized decorator doesn't take a class to be modified