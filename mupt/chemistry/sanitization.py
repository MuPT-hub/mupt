'''For automating RDKit molecule sanitization and chemical property cleanup'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Concatenate, Generator, Iterable, Optional, ParamSpec, Union
P = ParamSpec('P')
from functools import wraps

from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import SanitizeMol, SanitizeFlags, SANITIZE_ALL, SANITIZE_NONE
from rdkit.Chem.rdmolops import AromaticityModel, SetAromaticity, Kekulize

from ..mutils.decorators.functional import optional_in_place
from ..mutils.decorators.meta import extend_to_methods


@optional_in_place
def sanitize_mol(
        mol : Mol,
        sanitize_ops : Union[None, int, SanitizeFlags]=SANITIZE_NONE,
        aromaticity_model : Optional[AromaticityModel]=None,
    ) -> None:
    '''Apply chemical sanitization operations, including bond aromaticity
    By default, performs NO sanitization; all sanitization operations must be explicitly specified!
    '''
    # enforce correct typing from deliberate (or accidental) NoneType pass
    if sanitize_ops is None: 
        sanitize_ops = SANITIZE_NONE
    
    # aromaticity determination
    if aromaticity_model is not None:
        sanitize_ops = sanitize_ops & ~SanitizeFlags.SANITIZE_KEKULIZE & ~SanitizeFlags.SANITIZE_SETAROMATICITY # prevents final SanitizeMol call from undoing aromatcity model
        Kekulize(mol, clearAromaticFlags=True)
        SetAromaticity(mol, model=aromaticity_model)
        
    # hydrogen handling?
    ...
           
    # miscellaneous sanitization operations
    SanitizeMol(mol, sanitizeOps=sanitize_ops) # regardless of settings, sanitization should be done last to give greatest likelihodd of molecule validity

@extend_to_methods
def sanitizable_mol_outputs(mol_func : Callable[P, Union[Mol, Iterable[Mol]]]) -> Callable[Concatenate[Union[None, int, SanitizeFlags], Optional[AromaticityModel], P], Union[Mol, tuple[Mol, ...]]]:
    '''
    Decorator which injects molecule sanitization capability into a function which returns RDKit Mols
    By default, performs NO sanitization; all sanitization operations must be explicitly specified!
    
    Acts on functions which are assumed to return a single mol object, or an iterable containing Mols
    Decorated function will return a single mol in the former case, or a tuple of mols in the latter
    '''
    @wraps(mol_func)
    def wrapped_func(
            *args : P.args,
            sanitize_ops : Union[None, int, SanitizeFlags]=SANITIZE_NONE,
            aromaticity_model : Optional[AromaticityModel]=None,
            **kwargs : P.kwargs,
        ) -> Union[Mol, tuple[Mol, ...]]:
        # determine if return is singular, iterable, or invalid
        outputs = mol_func(*args, **kwargs)
        if isinstance(outputs, Mol):
            is_singular = True
            outputs = (outputs,)
        elif isinstance(outputs, Iterable):
            is_singular = False
            outputs = tuple(outputs) # cast to tuple in advance; all changes from here are made in-place
        else:
            raise TypeError(f'Expected wrapped function to return etiher a Chem.Mol object or an iterable of Chem.Mol, not {type(outputs)}')
        
        # perform sanitization
        for mol in outputs:
            if isinstance(mol, Mol): # for now, tolerates other types of objects in output stream by skipping over them
                sanitize_mol(mol, sanitize_ops=sanitize_ops, aromaticity_model=aromaticity_model, in_place=True)
                continue 

        return outputs[0] if is_singular else outputs
    return wrapped_func