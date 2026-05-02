'''Convenience utilities for drawing shaped objects'''

from typing import Any, Optional
from inspect import signature

from matplotlib.axes import Axes # DEV: eventually, declare explicit dependency on mpl...
from mpl_toolkits.mplot3d import Axes3D # ...(maybe even import conditionally?)
from matplotlib.pyplot import figure

from .shapes import BoundedShape


def visualize_shape(
    shape : BoundedShape,
    ax : Optional[Axes3D]=None,
    grid : bool=True, 
    **kwargs,
) -> Axes3D:
    '''Convenience interface for plotting a surface mesh for a class implementing the BoundedShape Protocol'''
    if ax is None:
        fig = figure()
        ax = fig.add_subplot(projection='3d')
    elif not isinstance(ax, Axes):
        raise TypeError(f'Require matplotlib Axes-like for shape mesh drawing, not {type(ax).__name__}')
    elif not isinstance(ax, Axes3D):
        raise ValueError('Provided Axes are not 3D, and cannot support shape mesh drawing')

    fig.set_tight_layout(True)
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