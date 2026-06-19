import numpy as np
import xarray as xr
from types import NoneType
from typing import Union, Callable
from ipywidgets import FloatSlider, HBox, VBox, interactive_output
from IPython.display import display
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.utils import create_sliders, create_sliders_from_dims

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.quiver import Quiver
from matplotlib.contour import QuadContourSet
from matplotlib.gridspec import GridSpec


font = {"family": "serif", "size": 12, "serif": "cmr10"}

matplotlib.rc("font", **font)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"

from bloch_schrodinger.plotting  import cmesh_tmpl, contour_tmpl, quiver_tmpl, get_template


def create_map_rescaling(
    fig: Figure,
    ax: Axes,
    dim1: str,
    dim2: str,
    data: xr.DataArray,
    method: str,
    template: dict = {},
    cst_bds:bool = False
    ) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in
        dim1 (str): coordinate along the x-axis of the plot
        dim2 (str): coordinate along the y-axis of the plot
        data (xr.DataArray): The data to plot.
        method (str): Which matplotlib 2D plot function to use between 'pcolormesh', 'contour' and 'contourf'.
        template (dict, optional): The template dictionnary contains all the instruction to create the plot. It has the following nested structure:
            template
                ↳ fkwargs: keyword arguments for the plotting function defined by 'method'. default to {}
                ↳ colorbar:
                    ↳ kwargs: keyword arguments passed to the Figure.colorbar function. default to {"format":"{x:.1e}"}.
                    ↳ cax: keyword arguments passed to the AxesDivider.append_axes function. Default to dict(position = 'right', size="5%", pad=0.05).
                    ↳ ticks: used to set manually the position of the colorbar ticks if necessary. Default to None.
                    ↳ tickslabel: used to set manually the text of the colorbar ticks if necessary. Default to None.
                ↳ slider_start: The initial position of the sliders. Default to 'left'.
                ↳ autoscale: Wheter to autoscale the color range. Default to True.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.
    """
    
    if method == 'pcolormesh':
        func = Axes.pcolormesh
    elif method == 'contour':
        func = Axes.contour
    elif method == 'contourf':
        func = Axes.contourf
            
    # Creating the sliders objects
    slider_dims = [dim for dim in data.dims if dim not in ["a1", "a2", dim1, dim2]]
    sliders = create_sliders_from_dims({dim:data.coords[dim] for dim in slider_dims}, start = template.get('slider_start', 'left'))
    
    # Creating the fkwargs key just in case, to avoid testing its existence every time
    if template.get('fkwargs') is None:
        template['fkwargs'] = {}
        
    # Initial parameter selections
    initial_field_sel = {dim: sliders[dim].value for dim in sliders}
    initial_coord_sel = {dim: sliders[dim].value for dim in sliders if dim in data.coords[dim1].dims}
    
    #Extracting the norm object from the template, it is convoluted to avoid unintended sharing of colorscales
    template = deepcopy(template) # We are going to mutate template so copying is important
    if template.get('fkwargs'):
        if template['fkwargs'].get('norm'):
            template['fkwargs']["norm"] = template['fkwargs']["norm"]() if callable(template['fkwargs']["norm"]) else template['fkwargs']["norm"]
    
    # Shortcut names for the coordinates
    X = data.coords[dim1].sel(initial_coord_sel)
    Y = data.coords[dim2].sel(initial_coord_sel)
    
    # Initial data selection
    plot_init = data.sel(initial_field_sel)

    obj = func(ax,
        X, Y, plot_init, **template['fkwargs']
    )
    
    if not cst_bds:
        ax.set_xlim(np.min(X), np.max(X))
        ax.set_ylim(np.min(Y), np.max(Y))
    
    if template.get('clim'):
        obj.set_clim(template['clim'][0], template['clim'][1])
    
    colorbar = template.get('colorbar')
    if colorbar is not None:
        if colorbar.get('kwargs') is None:
            colorbar['kwargs'] = {'format':"{x:.1e}"}
        
    if colorbar:
        divider = make_axes_locatable(ax)
        if colorbar.get('cax') is None:
            colorbar['cax'] = dict(position = 'right', size="5%", pad=0.05)
        cax = divider.append_axes(**colorbar['cax'])
        cbar = fig.colorbar(
            obj,
            cax=cax,
            **colorbar["kwargs"],
        )
        if colorbar.get("tickslabel"):
            ticks = colorbar.get("ticks", cbar.ax.get_yticks())
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(colorbar["tickslabel"])

    def update(**kwargs):
        
        nonlocal obj
        sel = {dim: kwargs[dim] for dim in sliders}
        
        field_sel = {
            dim: value
            for dim, value in sel.items()
        }

        coord_sel = {
            dim: value
            for dim, value in sel.items() if dim in data.coords[dim1].dims
        }
        
        newX = data.coords[dim1].sel(coord_sel, method="nearest")
        newY = data.coords[dim2].sel(coord_sel, method="nearest")
        new_plot = data.sel(field_sel, method="nearest")

        if hasattr(obj, "collections"):
            for coll in obj.collections:
                coll.remove()
        else:
            obj.remove()
            
        obj = func(ax,
            newX, newY, new_plot, **template['fkwargs']
        )
        
        if not cst_bds:
            ax.set_xlim(np.min(newX), np.max(newX))
            ax.set_ylim(np.min(newY), np.max(newY))

        
        if template.get("autoscale"):
            obj.autoscale()

        fig.canvas.draw_idle()

    return sliders, update, ax


def plot_eigenvector_rescaling(
    plots: list[list[Union[xr.DataArray, NoneType]]],
    potentials: list[list[Union[Potential, NoneType]]],
    templates: list[list[Union[str,tuple[Union[str, dict]]]]],
    quivers: Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]] = None,
    ncontours:int = 3,
    cst_bds: bool = False
    ) -> tuple[Figure, list[Axes]]:
    """The main function to plot eigenvectors in a interactive manner.

    Args:
        plots (list[list[xr.DataArray]]): A list of list of eigenvectors xr.DataArrays, each DataArray will be plotted in a separate subplot,
        in a grid-pattern determined by the structure of the list of lists.
        potentials (list[list[Union[Potential, NoneType]]]): The potentials to be plotted as contour for each plot.
        templates (list[list[Union[str,tuple[Union[str, dict]]]]]): A dictionnary containing all the instruction to define each subplot style. the templates can also be strings
        calling predefined templates, such as 'amplitude', 'phase', 'real' and more. see 'get_template' for more informations.
        titles (Union[NoneType, list[list[str]]): The title for each subplot. Default to None.
        quivers (Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]], optional): An optional argument to overlay quiver plots on top of the eigenvectors.
        Each entry of the list of lists must either be None or contain a tuple of DataArrays (U,V,C), see the quiver function from matplotlib for more informations.
        Defaults to None.
        ncontours (int, optional): The number of contours to use for the plot. This is a convenience argument. For more control, see the template formalism. Default to 3.
        cst_bds (bool, optional): Wheter to keep tight plot limits around the current grid scaling (False), of take the biggest grid possible from the get go and keep them constant (True). Default to False

    Raises:
        ValueError: Raise errors if the shapes are not consistent.
    """
    n_rows = len(plots)
    n_cols = len(plots[0])
    
    if len(templates) != n_rows or len(potentials) != n_rows:
        raise ValueError("different shapes for plots and templates")
    if len(templates[0]) != n_cols or len(potentials[0]) != n_cols:
        raise ValueError("different shapes for plots and templates")
    for i in range(1, n_rows):
        if len(plots[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'plots' not consistent")
        if len(templates[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'templates' not consistent")
        if len(potentials[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'potentials' not consistent")

    if quivers is None:
        quivers = [[None] * n_cols for u in range(n_rows)]

    funcs:list[Callable] = []
    sliders = {}
    
    def make_tmpl(template:Union[str, dict])->dict:
        """Check wheter template is a string or a dict, and if a str, create the proper dictionnary.
        """
        return template if isinstance(template, dict) else cmesh_tmpl(template)
    
    def format_template(template:tuple[Union[str, dict]])->tuple[dict, dict, dict]:
        """Format a template input into the proper tuple"""
        if isinstance(template, str):
            template = (cmesh_tmpl(template), contour_tmpl(ncontours), quiver_tmpl())
        elif isinstance(template, dict):
            template = (template, contour_tmpl(ncontours), quiver_tmpl())
        elif not isinstance(template, tuple):
            raise ValueError("Each template entry must either be a tuple or a string")
        elif len(template) == 1:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, contour_tmpl(ncontours), quiver_tmpl())
        elif len(template) == 2:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, template[1], quiver_tmpl())
        elif len(template) == 3:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, template[1], template[2])
        return template
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=(3 * (n_cols + 1), 3 * n_rows),
        layout="tight",
    )
    

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j]
            template = format_template(templates[i][j])
            
            plot = plots[i][j]
            poten = potentials[i][j]
            quiv = quivers[i][j]
            if plot is not None:
                slids, up, ax = create_map_rescaling(fig, ax, "x", "y", plot, "pcolormesh", template[0], cst_bds)
            sliders.update(slids)
            funcs += [up]
            if poten is not None:
                slids, up, ax = create_map_rescaling(fig, ax, "x", "y", poten.V, "contour", template[1], cst_bds)
            sliders.update(slids)
            funcs += [up]
            if quiv is not None:
                slids, up, ax = create_map_rescaling(fig, ax, "x", "y", quiv[0], quiv[1], template[2], cst_bds)
            sliders.update(slids)
            funcs += [up]
            
            if cst_bds:
                ax.set_xlim(np.min(plot.x), np.max(plot.x))
                ax.set_ylim(np.min(plot.y), np.max(plot.y))

            
            ax.set_aspect("equal")           
            axes[i][j] = ax
            
                        

    def update(**kwargs):
        for f in funcs:
            f(**kwargs)

    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
    return fig, axes    
