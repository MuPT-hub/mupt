'''Utilities for applying rigid transformations to other objects (not necessarily just points!)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Mapping, Sequence, Union
from typing import Protocol, runtime_checkable

from scipy.spatial.transform import RigidTransform


@runtime_checkable
class RigidTransformable(Protocol):
    '''Interface for objects that can undergo a rigid (isometric) transformation'''
    def apply_rigid_transformation(self, transformation: RigidTransform) -> Any: 
        # DEVNOTE: regarding typehints, returned type may be different to type of self, and is not necessarily transformable either
        ...

def apply_rigid_transformation_recursive(
        obj : Union[object, Sequence[Any], Mapping[str, Any]],
        transformation: RigidTransform,
    ) -> Union[object, Sequence[Any], dict[str, Any]]:
    '''Apply a rigid transformation to an object, if it supports such a transformation, and
    if the object is a Sequence or Mapping, attempt to transform its members recursively
    
    Parameters
    ----------
    obj : Any
        The object to be transformed, which may be a single object, a Sequence, or a Mapping
    rigid_transform : RigidTransform
        The rigid transformation to apply to the object

    Returns
    -------
    Any
        The transformed object, which (depending on the transformability and return types of
        the input and its members) may or many not be of the same type as the initial object
    '''
    # top-level application check
    if isinstance(obj, RigidTransformable):
        obj = obj.apply_rigid_transformation(transformation)

    # recursive iteration, as necessary
    if isinstance(obj, Sequence):  # DEVNOTE: specifically opted for Sequence over Iterable here to avoid double-covering Mappings and unpacking generators
        return type(obj)( # DEVNOTE: most common Sequence types (e.g. tuple, str, list) support init from comprehension; may revisit if this is not always the case
            apply_rigid_transformation_recursive(value, transformation)
                for value in obj
        ) 
    elif isinstance(obj, Mapping):
        return {
            key : apply_rigid_transformation_recursive(value, transformation)
                for (key, value) in obj.items()
        }
        
    return obj