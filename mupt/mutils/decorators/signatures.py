'''Tools for simplifying transfer and modification of wrapped function type signatures'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from inspect import Parameter, Signature

POSITIONAL_PARAMETER_TYPES = [
    Parameter.POSITIONAL_ONLY,
    Parameter.POSITIONAL_OR_KEYWORD,
    Parameter.VAR_POSITIONAL
]

def get_index_after_positionals(sig : Signature) -> int:
    '''Get the first Parameter index which follows all positional Parameters'''
    for i, param in enumerate(sig.parameters.values()):
        if param.kind not in POSITIONAL_PARAMETER_TYPES:
            return i + 1 # return index immediately after first non-positional argument
        else:
            return len(sig.parameters)

def insert_parameter_at_index(sig : Signature, new_param : Parameter, index : int) -> Signature:
    '''Insert a new Parameter into a Signature at a given position'''
    params = list(sig.parameters.values())
    params.insert(index, new_param)

    return sig.replace(parameters=params)

def modify_param_annotation_by_index(sig : Signature, index : int, new_type : type) -> Signature:
    '''Returns a copy of a Signature with the type annotation of the Parameter at a given index swapped out'''
    params = list(sig.parameters.values())
    old_param = params[index] # will raise IndexError if position is given; IT IS UP TO THE CALLER TO ENSURE THIS IS CORRECT!

    new_param = Parameter( # copy everything expect type annotation from old Parameter
        name=old_param.name,
        default=old_param.default,
        annotation=new_type,
        kind=old_param.kind
    )
    params[index] = new_param

    return sig.replace(parameters=params)

def modify_param_annotation_by_name(sig : Signature, param_name : str, new_type : type) -> Signature:
    '''Returns a copy of a Signature with the type annotation of the Parameter with a given name swapped out'''
    return modify_param_annotation_by_index(
        sig,
        index=list(sig.parameters.keys()).index(param_name), # look up location of target parameter
        new_type=new_type
    )