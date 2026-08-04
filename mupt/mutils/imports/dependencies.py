'''
Utilities to check for and enforce presence of
dependencies required by another piece of code
'''

from typing import Callable, Optional, ParamSpec, TypeVar, Union

Params = ParamSpec('Params')
ReturnType = TypeVar('ReturnType')

# from importlib import import_module
from importlib.util import find_spec
from functools import wraps
from .inspection import get_calling_module


class MissingPrerequisitePackage(Exception):
    '''
    Raised when a package dependency cannot be found
    and the caller should be alerted to install it
    '''
    def __init__(
        self,
        dependency_name : str,
        # DEV: should sentinels just be empty strings?
        dependency_name_formal : Optional[str]=None,
        use_case : str='import',
        install_link : Optional[str]=None,
        importing_package_name : Optional[str]=None,
    ):
        '''
        Supply information about the dependency to create a detailed error message

        Parameters
        ----------
        dependency_name : str
            The name of the module being imported, as appears in code (e.g. "numpy")
        dependency_name_formal : Optional[str], default None
            An optional stylized version of the module name, if different to how it appears in code
            E.g. "numpy" vs "NumPy", "sklearn" vs "Scikit-learn", etc.

            If Falsy value is provided, will default to dependency_name 
        use_case : str, default 'import'
            An optional specific reason the caller might want to import the dependent code
        install_link : Optional[str], default None
            An optional hyperlink to installation instructions for the dependency
        importing_package_name : Optional[str], default None
            An optional means of configuring which importing module is displayed in the error message
            Defaults to the module in which the caller is located, determined automatically 
        '''
        if not dependency_name_formal:
            dependency_name_formal = dependency_name

        if not importing_package_name:
            importing_package_name = get_calling_module().__spec__.name

        install_desc : str = f' by following the installation instructions at {install_link}' if install_link else ''
        
        message = (
            f'{use_case.capitalize()} require(s) {dependency_name_formal}, which was not found in the current context.'
            f'Please install `{dependency_name}`{install_desc}, then try importing from "{importing_package_name}" again'
        )
        
        super().__init__(message)
        
def module_installed(module_name : str) -> bool:
    '''
    Check whether a module of the given name is present on the system
    
    Parameters
    ----------
    module_name : str
        The name of the module, as it would occur in an import statement
        Do not support direct passing of module objects to avoid circularity 
        (i.e. no reason to check if a module is present if one has already imported it elsewhere)
    
    Returns
    -------
    module_found : bool
        Whether or not the module was found to be installed in the current working environment
    '''
    # try:
    #     package = import_module(module_name)
    # except ModuleNotFoundError:
    #     return False
    # else:
    #     return True
    
    try: # NOTE: opted for this implementation, as it never actually imports the package in question (faster and fewer side-effects)
        return find_spec(module_name) is not None
    except (ValueError, AttributeError, ModuleNotFoundError): # these could all be raised by a missing module
        return False
    
def modules_installed(*module_names : str) -> bool:
    '''
    Check whether one or more modules are all present
    Will only return true if ALL specified modules are found
    
    Parameters
    ----------
    module_names : *str
        Any number of module names, passed as a comma-separated sequence of strings
        
    Returns
    -------
    all_modules_found : bool
        Whether or not all modules were found to be installed in the current working environment
    '''
    return all(module_installed(module_name) for module_name in module_names)

def requires_modules(
    *required_module_names : str,
    missing_module_error : Union[Exception, type[Exception]]=ImportError,
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    '''
    Decorator which enforces optional module dependencies prior to function execution
    
    Parameters
    ----------
    module_names : *str
        Any number of module names, passed as a comma-separated sequence of strings
    missing_module_error : type[Exception], default ImportError
        The type of Exception to raise if a module is not found installed
        Defaults to ImportError
        
    Raises
    ------
    ImportError : Exception
        Raised if any of the specified packages is not found to be installed
        Exception message will indicate the name of the specific package found missing
    '''
    # meta-check to ensure type of raised Exception is valid
    if not isinstance(missing_module_error, Exception):
        if not (isinstance(missing_module_error, type) and issubclass(missing_module_error, Exception)):
            # DEV: this is potentially brittle, depending on how the specific Exception subtype is implemented?
            raise TypeError('Must pass either an Exception instance or an Exception subtype to "missing_module_error"')
    def tailored_exception(module_name : str) -> Exception:
        '''Accessory function to generate targetted Exceptions based on the provided
        mssing_module_error value and the name of a module with no found installation'''
        if isinstance(missing_module_error, Exception):
            return missing_module_error
        
        if isinstance(missing_module_error, type):
           return missing_module_error(f'No installation found for module "{module_name}"')
    
    def decorator(func : Callable[Params, ReturnType]) -> Callable[Params, ReturnType]:
        @wraps(func)
        def req_wrapper(*args : Params.args, **kwargs : Params.kwargs) -> ReturnType:
            for module_name in required_module_names:
                if not module_installed(module_name):
                    raise tailored_exception(module_name)
            else:
                return func(*args, **kwargs)
            
        return req_wrapper
    return decorator