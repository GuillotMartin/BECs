import numpy as np
import xarray as xr
from typing import Union, Type, Callable
import matplotlib.pyplot as plt
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.utils import create_sliders
from bloch_schrodinger.plotting import plot_cuts
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import VBox, interactive_output
from IPython.display import display
from copy import deepcopy
import warnings

class PotentialT(Potential):    
    def __init__(
        self,
        unitvecs: list[list[float, float]],
        resolution: tuple[int, int],
        v0: Union[int, float, complex, np.generic, xr.DataArray] = 100,
        dtype: Union[Type[int], Type[float], Type[complex], Type[np.generic]] = float,
    ):
        super().__init__(unitvecs, resolution, v0, dtype)

        self.coords: dict = {}  # Storing all the coordinates
        self.timefuncs: dict[
            dict[Callable, dict, dict]
        ] = {}  # Storing the functions of time building the potential
        self.shapes_t = {
            "V0": self.V
        }  # Storing the parts of the potential that are time-dependant.
        self.terms = {
            "V0": "V0"
        }  # Stores the terms of the time-dependant potential as string expressions to be evaluated
        self.make_time_ident()

    def copy(self):
        cop = super().copy()
        cop.coords = deepcopy(self.coords)
        cop.timefuncs = deepcopy(self.timefuncs)
        cop.shapes_t = deepcopy(self.shapes_t)
        cop.terms = deepcopy(self.terms)
        cop.make_time_ident()
        return cop

    def clear(self):
        super().clear()
        self.coords: dict = {}  # Storing all the coordinates
        self.timefuncs: dict[
            dict[Callable, dict, dict]
        ] = {}  # Storing the functions of time building the potential
        self.shapes_t = {
            "V0": self.V
        }  # Storing the parts of the potential that are time-dependant.
        self.terms = {
            "V0": "V0"
        }  # Stores the terms of the time-dependant potential as string expressions to be evaluated
        self.make_time_ident()

    def fromPotential(pot: Potential) -> "PotentialT":
        """Returns a PotentialT object constructed from a Potential object, with only a time independant part.

        Args:
            pot (Potential): The potential to convert
        """

        potT = PotentialT([pot.a1, pot.a2], pot.resolution, pot.v0, pot.dtype)
        potT.V = pot.V
        potT.update_V0()
        potT.x = pot.x
        potT.y = pot.y
        return potT

    def update_V0(self):
        """Update the time independant term of the potential"""
        self.coords.update(
            {dim: self.V.coords[dim] for dim in self.V.dims if dim not in ["a1", "a2"]}
        )
        self.shapes_t.update({"V0": self.V})

    def make_time_ident(self):
        """Add a simple time identity to the time functions"""
        self.timefuncs.update(
            {"t": {"func": lambda t: t, "dims": {}, "parameters": {}}}
        )

    def check_parameter(self, val: Union[float, xr.DataArray]) -> bool:
        """Check wheter a given parameter is a scalar value or an actual parameter dimension.
        If its a parameter dimension, this function will add it to the dimension dictionnary

        Args:
            val (Union[float, xr.DataArray]): The parameter to check

        Returns:
            bool: _description_
        """
        if isinstance(val, xr.DataArray):
            self.coords.update({val.name: val})
            return True
        else:
            return False

    def add_param(self, dictf: dict, name: str, val: Union[float, xr.DataArray]):
        """Small helper function to add a given parameter to the function dictionary.

        Args:
            dictf (dict): The dictionary of the function to update.
            name (str): The name of the parameter.
            val (Union[float, xr.DataArray]): the parameter to add.
        """
        if self.check_parameter(val):
            dictf["dims"].update({name: val})
        else:
            dictf["parameters"].update({name: val})

    def gaussian(
        self,
        name: str,
        t0: Union[float, xr.DataArray],
        sigma: Union[float, xr.DataArray],
        norm: str = "integral",
    ):
        """Construct a new time function following a gaussian shape.

        Args:
            name (str): The name of the time function for further use. must be unique.
            t0 (Union[float, xr.DataArray]): centrer time of the gaussian.
            sigma (Union[float, xr.DataArray]): Half-Width at Half Maximum of the gaussian.
            norm (str, optional): Normalization method, can either be 'integral' to get an integral equal to 1 or 'peak' to get a maximal value of 1. Defaults to 'integral'.
        """

        def fc(t, t0, sigma, norm=norm):
            norm = sigma / (2 * np.pi) ** 0.5 if norm == "integral" else 1
            return np.exp(-((t - t0) ** 2) / 2 / sigma**2) / norm

        dictfunc = {"func": fc, "dims": {}, "parameters": {}}

        self.add_param(dictfunc, "t0", t0)
        self.add_param(dictfunc, "sigma", sigma)

        self.timefuncs.update({name: dictfunc})

    def step(
        self,
        name: str,
        ts: Union[float, xr.DataArray],
        sigma: Union[float, xr.DataArray],
        vi: Union[float, xr.DataArray],
        vf: Union[float, xr.DataArray],
    ):
        """Construct a smooth step time function.

        Args:
            name (str): The name of the time function for further use. must be unique.
            ts (Union[float, xr.DataArray]): time at which the step occurs.
            sigma (Union[float, xr.DataArray]): width of the transition.
            vi (Union[float, xr.DataArray]): value before the step.
            vf (Union[float, xr.DataArray]): value after the step.
        """

        def fc(t, ts, sigma, vi, vf):
            steep = 5/(sigma + 1e-10)
            return vi + (vf - vi) * (1 / (1 + np.exp(-steep * (t - ts))))

        dictfunc = {"func": fc, "dims": {}, "parameters": {}}

        self.add_param(dictfunc, "ts", ts)
        self.add_param(dictfunc, "sigma", sigma)
        self.add_param(dictfunc, "vi", vi)
        self.add_param(dictfunc, "vf", vf)

        self.timefuncs.update({name: dictfunc})
        
    def sine(
        self,
        name: str,
        omega: Union[float, xr.DataArray] = 2*np.pi,
        phase: Union[float, xr.DataArray] = 0,
        amplitude: Union[float, xr.DataArray] = 1,
        mean: Union[float, xr.DataArray] = 0
    ):
        """Construct a sinusoidal modulation in time: amplitude * sin(omega*t + phase) + mean

        Args:
            name (str): The name of the time function for further use. must be unique.
            omega (Union[float, xr.DataArray], optional): reduced frequency of the oscillation. default to 2*pi.
            phase (Union[float, xr.DataArray], optional): phase of the oscillation. default to 0.
            amplitude (Union[float, xr.DataArray], optional): amplitude of the oscillation. default to 1.
            phase (Union[float, xr.DataArray], optional): mean value of the oscillation. default to 0.
        """

        def fc(t, omega, phase, amplitude, mean):
            return amplitude * np.sin(omega*t + phase) + mean

        dictfunc = {"func": fc, "dims": {}, "parameters": {}}

        self.add_param(dictfunc, "omega", omega)
        self.add_param(dictfunc, "phase", phase)
        self.add_param(dictfunc, "amplitude", amplitude)
        self.add_param(dictfunc, "mean", mean)

        self.timefuncs.update({name: dictfunc})

    def square(
        self,
        name: str,
        ti: Union[float, xr.DataArray],
        tf: Union[float, xr.DataArray],
        sigma: Union[float, xr.DataArray],
        vi: Union[float, xr.DataArray],
        vf: Union[float, xr.DataArray],
    ):
        """Construct a smooth quuare pulse time function.

        Args:
            name (str): The name of the time function for further use. must be unique.
            ti (Union[float, xr.DataArray]): time at which the pulse starts.
            tf (Union[float, xr.DataArray]): time at which the pulse stops.
            sigma (Union[float, xr.DataArray]): width of the transition.
            vi (Union[float, xr.DataArray]): value before the step.
            vf (Union[float, xr.DataArray]): value after the step.
        """

        def fc(t, ti, tf, sigma, vi, vf):
            steep = 5/(sigma + 1e-10)
            return vi + (vf - vi) * (1 / (1 + np.exp(-steep * (t - ti))) - 1 / (1 + np.exp(-steep * (t - tf))))

        dictfunc = {"func": fc, "dims": {}, "parameters": {}}

        self.add_param(dictfunc, "ti", ti)
        self.add_param(dictfunc, "tf", tf)
        self.add_param(dictfunc, "sigma", sigma)
        self.add_param(dictfunc, "vi", vi)
        self.add_param(dictfunc, "vf", vf)

        self.timefuncs.update({name: dictfunc})

    def ramp(
        self,
        name: str,
        ti: Union[float, xr.DataArray],
        tf: Union[float, xr.DataArray],
        vi: Union[float, xr.DataArray],
        vf: Union[float, xr.DataArray],
        smooth: Union[float, xr.DataArray],
    ):
        """Construct a smooth ramping function with slight overshoots.

        Args:
            name (str): The name of the time function for further use. must be unique.
            ts (Union[float, xr.DataArray]): time at which the step occurs.
            sigma (Union[float, xr.DataArray]): width of the transition.
            vi (Union[float, xr.DataArray]): value before the step.
            vf (Union[float, xr.DataArray]): value after the step.
        """

        def fc(t, ti, tf, vi, vf, smooth):
            tp1 = t - ti
            tp2 = tp1 - tf + ti
            f1 = tp1 / 2 * (1 + tp1 / (tp1**2 + smooth**2 + 1e-10) ** 0.5)
            f2 = tp2 / 2 * (1 + tp2 / (tp2**2 + smooth**2 + 1e-10) ** 0.5)

            return (f1 - f2) / (tf - ti) * (vf - vi) + vi

        dictfunc = {"func": fc, "dims": {}, "parameters": {}}

        self.add_param(dictfunc, "ti", ti)
        self.add_param(dictfunc, "tf", tf)
        self.add_param(dictfunc, "smooth", smooth)
        self.add_param(dictfunc, "vi", vi)
        self.add_param(dictfunc, "vf", vf)

        self.timefuncs.update({name: dictfunc})

    def create_timefunc(self, name: str, selection: dict) -> Callable:
        """Returns a lambda function based on the time function 'name' where the parameter dimensions have been set by a given selection.

        Args:
            name (str): The name of the time function to use
            selection (dict): The values selected for each of the parameters dimensions

        Returns:
            Callable: A lambda function with only a single argument 't'.
        """
        kwargs = {**self.timefuncs[name]["parameters"]}
        for param in self.timefuncs[name]["dims"]:
            dims = self.timefuncs[name]["dims"][param].dims # The dimensions of the parameter array 'param
            vals = self.timefuncs[name]["dims"][param]
            # print(f"dim: {dims}, vals:{vals}, param:{param}, selection: {selection}")
            kwargs.update({param: vals.sel({dim:selection[dim] for dim in dims}, method = 'nearest').data}) # added nearest option to avoid some rounding errors
        return lambda t: self.timefuncs[name]["func"](t, **kwargs)

    def plot_timefunction(
        self, name: Union[str, list[str]], tmin: float, tmax: float, n_t: int = 100
    ) -> tuple[Figure, Axes]:
        """Create an interactive plot showing a time function with its parameters as sliders. Can plot multiple time functions at the same time.

        Args:
            name (Union[str, list[str]]): The name (names) of the time function(s) to plot.
            tmin (float): lower bound for the plotting window.
            tmax (float): Upper bound for the plotting window.
            n_t (int, optional): Number of points to compute. Defaults to 100.

        Returns:
            tuple[Figure, Axes]: Figure and axes objects to be plotted.
        """
        t = create_parameter("t", np.linspace(tmin, tmax, n_t))

        if isinstance(name, str):
            arr = self.timefuncs[name]["func"](
                t, **self.timefuncs[name]["parameters"], **self.timefuncs[name]["dims"]
            ).squeeze()
            
            return plot_cuts(
                arr,
                "t",
                groupby=[],
            )

        elif isinstance(name, list):
            arrs = []
            ndim = create_parameter("funcs", np.arange(len(name)))
            for na in name:
                sub_arr = self.timefuncs[na]["func"](
                    t, **self.timefuncs[na]["parameters"], **self.timefuncs[na]["dims"]
                )
                arrs += [sub_arr]

            arr = xr.concat(arrs, dim=ndim, coords='minimal')
            fig, ax = plot_cuts(
                arr.squeeze(),
                "t",
                groupby=["funcs"],
            )

            ax.legend(name)
            return fig, ax

        else:
            raise ValueError("'name' must be a list or a string.")

    def add_shape(self, name: str, shape: xr.DataArray):
        """Add a time-dependant data array to the object.

        Args:
            name (str): Name of the data.
            shape (xr.DataArray): Time dependant part of the potential, can have multiple dimensions, and needs at least to have 'a1' and 'a2'.
        """

        if "a1" not in shape.dims or "a2" not in shape.dims:
            raise ValueError(
                "The shape given does not have the dimensions 'a1' or 'a2'"
            )
        self.coords.update(
            {dim: shape.coords[dim] for dim in shape.dims if dim not in ["a1", "a2"]}
        )
        self.shapes_t.update({name: shape})

    def circle_t(
        self,
        name: str,
        center: tuple[Union[float, xr.DataArray]],
        radius: Union[float, xr.DataArray],
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 1,
    ):
        """Add 'value' to the potential in a circle. Support coordinates attribution for all parameters.

        Args:
            name (str): The name of this time-dependant feature.
            center (tuple[Union[float,xr.DataArray]]): The center of the circle in the x,y basis.
            radius (Union[float,xr.DataArray]): The radius of the circle
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the circle. Defaults to 1.
        """
        r = ((self.V.x - center[0]) ** 2 + (self.V.y - center[1]) ** 2) ** 0.5

        v1 = value
        v2 = 0

        if inverse:
            v1 = 0
            v2 = value

        shape = xr.where(r < radius, v1, v2)
        self.add_shape(name, shape)

    def rectangle_t(
        self,
        name: str,
        center: tuple[Union[float, xr.DataArray]],
        dims: tuple[Union[float, xr.DataArray]],
        rotation: Union[float, xr.DataArray] = 0,
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 1,
    ):
        """Add 'value' to the potential in a rectangle. Support coordinates attribution for all parameters.

        Args:
            name (str): The name of this time-dependant feature.
            center (tuple[Union[float,xr.DataArray]]): The center of the rectangle in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The length along x and y.
            rotation (Union[float,xr.DataArray]): A rotation (in radians) of the rectangle. default to 0.
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 1.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        x = self.V.x - center[0]
        y = self.V.y - center[1]

        x_rot = +x * xr.ufuncs.cos(rotation) + y * xr.ufuncs.sin(rotation)
        y_rot = +x * xr.ufuncs.sin(rotation) - y * xr.ufuncs.cos(rotation)

        v1 = value
        v2 = 0

        if inverse:
            v1 = 0
            v2 = value

        shape = xr.where(
            (abs(x_rot) < dims[0] / 2) * (abs(y_rot) < dims[1] / 2), v1, v2
        )
        self.add_shape(name, shape)

    def ellipse_t(
        self,
        name: str,
        center: tuple[Union[float, xr.DataArray]],
        dims: tuple[Union[float, xr.DataArray]],
        rotation: Union[float, xr.DataArray] = 0,
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 1,
    ):
        """Add 'value' to the potential in an ellipse. Support coordinates attribution for all parameters.

        Args:
            name (str): The name of this time-dependant feature.
            center (tuple[Union[float,xr.DataArray]]): The center of the ellipse in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The semi-axes along x and y.
            rotation (Union[float,xr.DataArray]): A rotation (in radians) of the ellipse axis. default to 0.
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 1.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        x = self.V.x - center[0]
        y = self.V.y - center[1]

        x_rot = +x * xr.ufuncs.cos(rotation) + y * xr.ufuncs.sin(rotation)
        y_rot = +x * xr.ufuncs.sin(rotation) - y * xr.ufuncs.cos(rotation)

        r = ((x_rot / dims[0]) ** 2 + (y_rot / dims[1]) ** 2) ** 0.5

        v1 = value
        v2 = 0

        if inverse:
            v1 = 0
            v2 = value

        shape = xr.where(r < 1, v1, v2)
        self.add_shape(name, shape)

    def modulate(self, expression: str):
        """A high level function to modulate the time-independant potential V0 by a time function.

        Args:
            expression (str): The modulation expression as a string to be evaluated
        """
        self.terms["V0"] = f"V0 * {expression}"

    def add_term(self, expression: str, name: str = None, duplicate: bool = False):
        """Add a term to the time-dependant potential. A term is an analytical expression made of shapes and time functions.

        Args:
            expression (str): A string representing the expression of the term, can contain numpy function, shortened as 'np'.
            name (str, optional): Name of the term, useful for separate visualization, if no name is given, a unique id will be given as a name.
            duplicate (bool, optional): Wheter to duplicate the expression if it is already in 'terms'. Default to False.
        """
        if name is None:
            name = str(len(self.terms))
        if expression not in self.terms.values() or duplicate:
            self.terms.update({name: expression})
        else:
            warnings.warn("The expression is already in terms, and was not duplicated")

    def to_potential(
        self,
        t: float = None,
        t_coord: Union[tuple[float, float, int], xr.DataArray] = None,
    ) -> Potential:
        """Return a Potential object evaluated at a specified time t or with a dimension t. Useful for plotting functions.

        Args:
            t (float): If not None, then the potential returned is evaluated at the time t. Default to None.
            t_coord (Union[tuple[float,float,int], xr.DataArray]): Specify the time dimensions to add, can either be a tuple (tmin, tmax, n_points)
            to create a linear array, or directly a time coordinate xarray. Default to None.

        Returns:
            Potential: A Potential object.
        """

        pot = Potential([self.a1, self.a2], self.resolution)

        self.update_V0()
        context = {**self.shapes_t}
        Vtmp = 0

        if t is not None:
            for func in self.timefuncs:
                arr = self.timefuncs[func]["func"](
                    t,
                    **self.timefuncs[func]["parameters"],
                    **self.timefuncs[func]["dims"],
                )
                context.update({func: arr})

            for term in self.terms.values():
                Vtmp = Vtmp + eval(term, {"__builtins__": {}}, context)

        else:
            if isinstance(t_coord, tuple):
                t_coord = create_parameter(
                    "t", np.linspace(t_coord[0], t_coord[1], t_coord[2])
                )

            for func in self.timefuncs:
                arr = self.timefuncs[func]["func"](
                    t_coord,
                    **self.timefuncs[func]["parameters"],
                    **self.timefuncs[func]["dims"],
                )
                context.update({func: arr})

            for term in self.terms.values():
                Vtmp = Vtmp + eval(term, {"__builtins__": {}}, context)

        pot.V = Vtmp
        return pot

    def plot_t(
        self, tmin: float, tmax: float, n_t: int = 100, **kwargs
    ) -> tuple[Figure, Axes]:
        """Creates an interactive plot of the potential, with all the parameters as sliders. Must be used in an interactive python session, preferably a notebook.
        kwargs are passed to the matplotlib pcolormesh function."""

        t = create_parameter("t", np.linspace(tmin, tmax, n_t))
        self.update_V0()
        context = {**self.shapes_t}

        Vtmp = 0
        for func in self.timefuncs:
            arr = self.timefuncs[func]["func"](
                t, **self.timefuncs[func]["parameters"], **self.timefuncs[func]["dims"]
            )
            context.update({func: arr})

        for term in self.terms.values():
            Vtmp = Vtmp + eval(term, {"__builtins__": {}}, context)

        Vtmp = Vtmp.squeeze()
               
        slider_dims = [dim for dim in Vtmp.dims if dim not in ["a1", "a2", "x", "y"]]
        sliders = create_sliders(Vtmp, slider_dims)

        initial_sel = {dim: sliders[dim].value for dim in slider_dims}
        potential = Vtmp.sel(initial_sel)

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(Vtmp.x, Vtmp.y, potential, shading="auto", **kwargs)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(
            mesh,
            cax=cax,
            label="Potential",
        )

        cbar.set_label("Potential")

        def update(**slkwargs):
            sel = {dim: slkwargs[dim] for dim in slider_dims}

            new_potential = Vtmp.sel(sel, method="nearest")
            mesh.set_array(new_potential.data.reshape(-1))
            mesh.set_clim(
                vmin=kwargs.get("vmin", float(new_potential.min())),
                vmax=kwargs.get("vmax", float(new_potential.max())),
            )
            ax.set_title(", ".join([f"{d}={sel[d]:.3f}" for d in sel]))
            fig.canvas.draw_idle()

        out = interactive_output(update, sliders)
        # Display everything
        display(VBox(list(sliders.values()) + [out]))
        return fig, ax

    def make_Vt(self, selection: dict[float]) -> Callable:
        """The main point of this class. Given a selection of value for each parameter dimension,
        return a function who takes as input the time and returns the potential landscape at this time.
        This function is very fast and intended for evaluation during time-dependant simulations.

        Args:
            selection (dict[float]): The selection of parameters value.

        Returns:
            Callable: The function V(t) given the selection of parameter values.
        """
        self.update_V0()
        funcs_sel = {}
        for func in self.timefuncs:
            funcs_sel.update({func: self.create_timefunc(func, selection)})

        shape_sel = {}
        for name, shape in self.shapes_t.items():
            subsel = {dim: selection[dim] for dim in selection if dim in shape.dims}
            shape_sel.update({name: shape.sel(subsel)})

        context = {"np": np, **shape_sel}
        

        def Vt(t: float):
            for nfunc, func in funcs_sel.items():
                context.update({nfunc: func(t)})

            pot = 0
            for term in self.terms.values():
                pot = pot + eval(term, {"__builtins__": {}}, context)
            return pot

        return Vt


from bloch_schrodinger.potential import create_parameter  # noqa: E402
if __name__ == "__main__":
    foo = PotentialT(
        [[2, 0], [0, 2]],
        (1024, 1024),
    )

    foo.rectangle(center=(-0.5, -0.5), dims=(0.5, 0.7))

    bar = create_parameter("t_pulse", np.linspace(1, 3, 3))
    bar2 = create_parameter("sigma_pulse", np.linspace(1, 2, 2))

    bar3 = create_parameter("width_step", np.linspace(0, 1, 4))
    bar4 = create_parameter("vf", np.linspace(1, 2, 2))

    foo.gaussian("pulse", bar, bar2, norm="peak")
    foo.ramp("step", 0, 3, 0, bar4, bar3)

    # foo.plot_timefunction(['step', 'pulse'], -2, 10)
    # plt.show()
    bar5 = create_parameter("xcirc", np.linspace(-0.5, 0.5, 2))

    foo.circle_t("circle", center=(bar5, 0), radius=0.5, value=-40)

    foo.ellipse_t("ellipse", center=(0.3, 0), dims=(0.3, 0.4), value=-100)

    foo.add_term("circle * pulse")
    foo.add_term("ellipse * step")

    selection = {
        "t_pulse": 2,
        "sigma_pulse": 1,
        "width_step": 2,
        "vf": 1,
    }

    Vt = foo.make_Vt(selection)

    foopot = Potential(
        [[2, 0], [0, 2]],
        (128, 128),
    )

    foopot.rectangle(center=(bar5, -0.5), dims=(0.5, 0.7))

    foo2 = PotentialT.fromPotential(foopot)

    foo2.plot_t(0, 2)
