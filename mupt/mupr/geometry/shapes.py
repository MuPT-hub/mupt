'''For encoding rigid bodies in space'''

from typing import Generic, TypeVar

T = TypeVar('T')

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ...mutils.decorators.classmod import register_abstract_class_attrs
from ...mutils.arraytypes import Shape, DType, M, N, Dims, Vector3, ArrayNxN


@register_abstract_class_attrs('dimension') # requires that subclasses implement a dimensionality at the class level
class BoundedShape(ABC, Generic[T]): # template for numeric type (some iterations of float in most cases)
    '''Interface for bounded, rigid bodies'''
    @property
    @abstractmethod
    def centroid(self) -> np.ndarray[Shape[Dims], T]:
        '''Coordinate of the geometric center of the body'''
        ...
    # COM = CoM = center_of_mass = centroid # aliases for convenience
    
    @property
    @abstractmethod
    def volume(self) -> T:
        '''Cumulative measure within the boundary of the body'''
        ...
        
    @abstractmethod
    def contains(self, point : np.ndarray[Shape[Dims], T]) -> bool:
        '''Whether a given coordinate lies within the boundary of the body'''
        ... 
        
    @abstractmethod
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[N, N], T]) -> 'BoundedShape':
        '''Implemenation of how the body should actually apply an affine transformation matrix'''
        ...

    def affine_transformation(self, affine_matrix : np.ndarray[Shape[N, N], T]) -> 'BoundedShape':
        '''
        Apply an affine transformation to the body, as encoded by a transformation matrix
        Matrix should be square and have dimension exactly one greater that that of the body
        '''
        assert(affine_matrix.shape == (self.dimension + 1, self.dimension + 1)) # enforce squareness and dimensionality of transform
        return self._apply_affine_transformation(affine_matrix)
     
     # TODO : require definition of support function?

# Concrete BoundedShape implementations
@dataclass
class Sphere(BoundedShape[float], dimension=3):
    '''A spherical body with arbitrary radius and center'''
    radius : float
    center : np.ndarray[Shape[3], float]
    
    @property
    def centroid(self) -> np.ndarray[Shape[3], float]:
        return self.center
    
    @property
    def volume(self) -> float:
        return 4/3 * np.pi * self.radius**3
    
    def contains(self, point : np.ndarray[Shape[3], float]) -> bool:   # TODO: decide whether containment should be boundary-inclusive
        return bool(np.linalg.norm(self.center - point) < self.radius) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], T]) -> 'Sphere':
        affine_center = np.concatenate([self.center, [1.0]]) # center point in homogeneous coordinates
        transformed_center_homog = affine_matrix @ affine_center
        transformed_center = transformed_center_homog[:-1] / transformed_center_homog[-1] # project back down from homogeneous coordinates
        
        return Sphere( # TODO: should return an ellipsoid if scaling in anisotropic
            radius=self.radius * np.linalg.det(affine_matrix)**(1/3), # scale radius appropriately
            center=transformed_center,
        )

