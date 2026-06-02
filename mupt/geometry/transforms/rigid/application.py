'''Utilities for applying rigid transformations to other objects (not necessarily just points!)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Mapping, Self, Sequence, Union
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod

from scipy.spatial.transform import RigidTransform

from ....mutils.copyable import Copyable, NotCopyableError


@runtime_checkable
class RigidlyTransformable(Protocol):
    '''Mixin for objects which support rigid transformations'''
    # DEV: went back and forth on verbiage, but settled on the following as least ambiguous:
    # * "transformation" to refer to the RigidTransforms passed around
    # * "transform" to refer to the act of applying a transformation (NOT the transformation itself)
    # * "transformed" to indicate that a transform was applied to a COPY of an object
    # DON'T change these names until you've understood this and made similar considerations for proposed changes

    # transform provenance
    @property
    def cumulative_transformation(self) -> RigidTransform:
        '''
        The net transformation applied to this object since it was instantiated
        Heuristically, this describes "how much the object has been moved from where it started"

        Used for tracking provenance of rigid transformations and handing that information off between objects
        '''
        if not hasattr(self, '_cumul_transf'): # DEV: deliberately mangled name to not resemble it "public" counterpart much
            self._cumul_transf = RigidTransform.identity()
        return self._cumul_transf
    
    @cumulative_transformation.setter
    def cumulative_transformation(self, transformation : RigidTransform) -> None:
        # DEV: might include some additional checks in here in the future
        self._cumul_transf = transformation
        
    @property
    def resetting_transformation(self) -> RigidTransform:
        '''The transformation that resets the object to its original configuration'''
        return self.cumulative_transformation.inv()

    # in-place application of transformations
    @abstractmethod
    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        raise NotImplementedError # implement subclass-specific behavior here
        
    def rigidly_transform(self, transformation : RigidTransform) -> None:
        '''Apply a rigid transformation to this object in-place'''
        self._rigidly_transform(transformation)
        self.cumulative_transformation = transformation * self.cumulative_transformation
    
    def reset_transform(self) -> None:
        '''Return the object to its un-transformed configuration'''
        self.rigidly_transform(self.resetting_transformation)

    # copying and out-of-place applications of transformations

    ## DEV: _copy_untransformed() is deliberately NOT an abstract method, as it's not required that child classes implement it;
    ## ...if children don't implement it, they simply won't be able to perform copying or out-of-place transformations
    def _copy_untransformed(self) -> Self:
        '''Defines how to make a copy of an object with the same internal parts, but  without preserving it's cumulative transformation'''
        raise NotCopyableError(f'Class {self.__class__.__name__} does not define how instances should copy their parts')

    def copy(self) -> Self:
        '''Make a copy of this BoundedShape object, with transformation history preserved'''
        new_obj = self._copy_untransformed()
        new_obj.cumulative_transformation = self.cumulative_transformation # transfer net displacement WITHOUT directly transforming copied parts
        
        return new_obj

    def rigidly_transformed(self, transformation: RigidTransform) -> Self:
        '''Return a copy of this object which has been transformed according to the rigid transformation provided'''
        clone = self.copy() # TODO: implement mechanism to transfer cumul transform during copy of child classes
        clone.rigidly_transform(transformation)
        
        return clone

    def reset_transformed(self) -> Self:
        '''Return an un-transformed copy of this object'''
        return self.rigidly_transformed(self.resetting_transformation)
        
        
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
    if isinstance(obj, RigidlyTransformable):
        obj = obj.rigidly_transformed(transformation)

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