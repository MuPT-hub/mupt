'''Convenience utilities for drawing shaped objects'''

from typing import Any, Optional, TYPE_CHECKING
from inspect import signature

from .shapes import BoundedShape

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def visualize_shape(
    shape : BoundedShape,
    ax : Optional['Axes3D']=None,
    grid : bool=True, 
    **kwargs,
) -> 'Axes3D':
    '''Convenience interface for plotting a surface mesh for a class implementing the BoundedShape Protocol'''
    # DEV TODO: eventually, have a more standardized way to deal with these kinds of dependencies on import (a la @requires(...))
    try:
        from matplotlib.axes import Axes
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.pyplot import figure
    except ImportError as exc:
        raise RuntimeError(
            'matplotlib is required to use visualize_shape(); install matplotlib to enable shape visualization'
        ) from exc

    if ax is None:
        fig = figure(layout='tight')
        ax = fig.add_subplot(projection='3d')
    elif not isinstance(ax, Axes):
        raise TypeError(f'Require matplotlib Axes-like for shape mesh drawing, not {type(ax).__name__}')
    elif not isinstance(ax, Axes3D):
        raise ValueError('Provided Axes are not 3D, and cannot support shape mesh drawing')

    ax.set_autoscale_on(True)
    if grid:
        ax.set_axis_on()
    else:
        ax.set_axis_off()

    valid_mesh_kws : set[str] = signature(shape.surface_mesh).parameters.keys() - {'self'}
    mesh_kwargs : dict[str, Any] = {
        kw : kwargs.pop(kw)
            for kw in valid_mesh_kws
                if kw in kwargs
    }

    vertices, triangles = shape.surface_mesh(**mesh_kwargs)
    ax.plot_trisurf(*vertices.T, triangles=triangles, **kwargs)
    ax.set_title(type(shape).__name__)
    ax.set_aspect('equal') # avoids scaling distortion along axes (otherwise, Ellipsoids would plot like Spheres)
    
    return ax