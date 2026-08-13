import warnings
from collections.abc import Callable

import numpy as np
import xarray as xr
from bloch_schrodinger.plotting import plot_cuts
from bloch_schrodinger.potential import Potential, create_parameter
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class ParametricPotential(Potential):
    """The machinery shared by the two time-dependent potential classes of this module.

    A parametric potential is described by two things:

    - a registry of named functions (``self.funcs``), each stored as
      ``{"func": callable, "dims": {...}, "parameters": {...}}``. A parameter of such a function is
      either a plain value, kept in "parameters", or a whole parameter dimension wrapped in a
      DataArray, kept in "dims" and registered in ``self.param_coords`` so that the solvers know
      they have to iterate over it.
    - a dictionary of string expressions (``self.terms``), combining those functions and whatever
      else the subclass puts in the evaluation context into the final potential.

    The subclasses differ only in the signature of the functions they register: PotentialT registers
    pure time functions ``f(t, **parameters)``, while AnalyticPotential registers functions of time
    and space ``f(t, *coords, **parameters)``. The two hooks '_wrap_timefunc' and '_call_func'
    absorb that difference, so that everything below is written only once.
    """

    def __init__(
        self,
        unitvecs: list[list[float]],
        resolution: tuple[int],
        v0: complex | np.generic | xr.DataArray = 0,
        dtype: type[int] | type[float] | type[complex] | type[np.generic] = float,
        endpoint: bool = False,
    ):
        """Initialize a parametric potential over a unit cell, see the Potential class for the
        meaning of the arguments. As for a Potential, the dimension of the described space (1D, 2D
        or 3D) is set by the number of unit vectors given.
        """
        super().__init__(unitvecs, resolution, v0, dtype, endpoint)

        self.spatial_dims = [f"a{i + 1}" for i in range(self.n_dims)]
        self.param_coords: dict[str, xr.DataArray] = {}  # Storing all the parameter dimensions
        self.funcs: dict[str, dict] = {}  # Storing the functions building the potential
        self.terms: dict[str, str] = {}  # The expressions to be evaluated, made of those functions
        self.make_time_ident()

    def clear(self):
        """Remove all parameter dimensions, functions and terms from the potential"""
        super().clear()
        self.param_coords = {}
        self.funcs = {}
        self.terms = {}
        self.make_time_ident()

    ### ====================
    ### Subclass hooks
    ### ====================

    def _wrap_timefunc(self, fc: Callable) -> Callable:
        """Adapt a pure time function f(t, **parameters) to the signature this class registers.

        Args:
            fc (Callable): The time function to adapt.

        Returns:
            Callable: The function, as it should be stored in the registry.
        """
        return fc

    def _call_func(self, func: Callable, t: float | xr.DataArray, coords: list, kwargs: dict):
        """Evaluate a registered function, passing it the coordinates if this class needs them.

        Args:
            func (Callable): The function to evaluate.
            t (Union[float,xr.DataArray]): The time, or a whole time coordinate.
            coords (list): The spatial coordinates, one array per dimension. Unused here.
            kwargs (dict): The parameters of the function.
        """
        return func(t, **kwargs)

    ### ====================
    ### Parameters and functions
    ### ====================

    def check_parameter(self, val: float | xr.DataArray) -> bool:
        """Check whether a given parameter is a scalar value or an actual parameter dimension.
        If it is a parameter dimension, this function will add it to the dimension dictionnary.

        Args:
            val (Union[float, xr.DataArray]): The parameter to check

        Returns:
            bool: True if the parameter is a parameter dimension.
        """
        if isinstance(val, xr.DataArray):
            self.param_coords.update({val.name: val})
            return True
        else:
            return False

    def add_param(self, dictf: dict, name: str, val: float | xr.DataArray):
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

    def add_function(
        self,
        name: str,
        function: Callable,
        parameters: dict[str, float | xr.DataArray] | None = None,
    ):
        """Add a function to the registry, for use by the term expressions.

        Args:
            name (str): The name of the function, as it will be called in the term expressions.
            Must be unique.
            function (Callable): The function to register. It takes the time as first argument,
            then, for an AnalyticPotential only, one argument per spatial dimension, then its own
            parameters as keywords.
            parameters (dict[Union[float,xr.DataArray]], optional): A dictionnary containing the
            value of each additional parameter of the function. The parameters can either take
            scalar or DataArray values. Defaults to no parameter.
        """
        dictfunc = {"func": function, "dims": {}, "parameters": {}}

        for key, value in (parameters or {}).items():
            self.add_param(dictfunc, key, value)

        self.funcs.update({name: dictfunc})

    def make_time_ident(self):
        """Register the identity time function 't', always available to the term expressions"""
        self.add_function("t", self._wrap_timefunc(lambda t: t))

    def create_func(self, name: str, selection: dict) -> Callable:
        """Returns a function based on the registered function 'name', where the parameter
        dimensions have been fixed by a given selection.

        Args:
            name (str): The name of the function to use
            selection (dict): The values selected for each of the parameter dimensions

        Returns:
            Callable: The function, with only its time (and, for an AnalyticPotential, its
            coordinates) left as arguments.
        """
        kwargs = {**self.funcs[name]["parameters"]}
        for param, values in self.funcs[name]["dims"].items():
            # 'nearest' avoids some rounding errors on the selected values
            kwargs.update(
                {
                    param: values.sel(
                        {dim: selection[dim] for dim in values.dims}, method="nearest"
                    ).data
                }
            )
        func = self.funcs[name]["func"]
        return lambda *args: func(*args, **kwargs)

    def eval_func(
        self, name: str, t: float | xr.DataArray, coords: list | None = None
    ) -> xr.DataArray:
        """Evaluate a registered function over the whole parameter space, keeping its parameter
        dimensions as dimensions of the result.

        Args:
            name (str): The name of the function to evaluate.
            t (Union[float,xr.DataArray]): The time, or a whole time coordinate.
            coords (list, optional): The spatial coordinates to evaluate the function on, one array
            per dimension. Only used by AnalyticPotential, which defaults to the potential's own grid.

        Returns:
            xr.DataArray: The evaluated function.
        """
        entry = self.funcs[name]
        return self._call_func(
            entry["func"], t, coords, {**entry["parameters"], **entry["dims"]}
        )

    ### ====================
    ### Time functions
    ### ====================

    def gaussian(
        self,
        name: str,
        t0: float | xr.DataArray,
        sigma: float | xr.DataArray,
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

        self.add_function(name, self._wrap_timefunc(fc), {"t0": t0, "sigma": sigma})

    def step(
        self,
        name: str,
        ts: float | xr.DataArray,
        sigma: float | xr.DataArray,
        vi: float | xr.DataArray,
        vf: float | xr.DataArray,
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
            steep = 5 / (sigma + 1e-10)
            return vi + (vf - vi) * (1 / (1 + np.exp(-steep * (t - ts))))

        self.add_function(
            name, self._wrap_timefunc(fc), {"ts": ts, "sigma": sigma, "vi": vi, "vf": vf}
        )

    def sine(
        self,
        name: str,
        omega: float | xr.DataArray = 2 * np.pi,
        phase: float | xr.DataArray = 0,
        amplitude: float | xr.DataArray = 1,
        mean: float | xr.DataArray = 0,
    ):
        """Construct a sinusoidal modulation in time: amplitude * sin(omega*t + phase) + mean

        Args:
            name (str): The name of the time function for further use. must be unique.
            omega (Union[float, xr.DataArray], optional): reduced frequency of the oscillation. default to 2*pi.
            phase (Union[float, xr.DataArray], optional): phase of the oscillation. default to 0.
            amplitude (Union[float, xr.DataArray], optional): amplitude of the oscillation. default to 1.
            mean (Union[float, xr.DataArray], optional): mean value of the oscillation. default to 0.
        """

        def fc(t, omega, phase, amplitude, mean):
            return amplitude * np.sin(omega * t + phase) + mean

        self.add_function(
            name,
            self._wrap_timefunc(fc),
            {"omega": omega, "phase": phase, "amplitude": amplitude, "mean": mean},
        )

    def square(
        self,
        name: str,
        ti: float | xr.DataArray,
        tf: float | xr.DataArray,
        sigma: float | xr.DataArray,
        vi: float | xr.DataArray,
        vf: float | xr.DataArray,
    ):
        """Construct a smooth square pulse time function.

        Args:
            name (str): The name of the time function for further use. must be unique.
            ti (Union[float, xr.DataArray]): time at which the pulse starts.
            tf (Union[float, xr.DataArray]): time at which the pulse stops.
            sigma (Union[float, xr.DataArray]): width of the transition.
            vi (Union[float, xr.DataArray]): value before the step.
            vf (Union[float, xr.DataArray]): value after the step.
        """

        def fc(t, ti, tf, sigma, vi, vf):
            steep = 5 / (sigma + 1e-10)
            return vi + (vf - vi) * (
                1 / (1 + np.exp(-steep * (t - ti))) - 1 / (1 + np.exp(-steep * (t - tf)))
            )

        self.add_function(
            name,
            self._wrap_timefunc(fc),
            {"ti": ti, "tf": tf, "sigma": sigma, "vi": vi, "vf": vf},
        )

    def ramp(
        self,
        name: str,
        ti: float | xr.DataArray,
        tf: float | xr.DataArray,
        vi: float | xr.DataArray,
        vf: float | xr.DataArray,
        smooth: float | xr.DataArray,
    ):
        """Construct a smooth ramping function with slight overshoots.

        Args:
            name (str): The name of the time function for further use. must be unique.
            ti (Union[float, xr.DataArray]): time at which the ramp starts.
            tf (Union[float, xr.DataArray]): time at which the ramp stops.
            vi (Union[float, xr.DataArray]): value before the ramp.
            vf (Union[float, xr.DataArray]): value after the ramp.
            smooth (Union[float, xr.DataArray]): smoothing of the two corners of the ramp.
        """

        def fc(t, ti, tf, vi, vf, smooth):
            tp1 = t - ti
            tp2 = tp1 - tf + ti
            f1 = tp1 / 2 * (1 + tp1 / (tp1**2 + smooth**2 + 1e-10) ** 0.5)
            f2 = tp2 / 2 * (1 + tp2 / (tp2**2 + smooth**2 + 1e-10) ** 0.5)

            return (f1 - f2) / (tf - ti) * (vf - vi) + vi

        self.add_function(
            name,
            self._wrap_timefunc(fc),
            {"ti": ti, "tf": tf, "vi": vi, "vf": vf, "smooth": smooth},
        )

    ### ====================
    ### Terms
    ### ====================

    def add_term(self, expression: str, name: str | None = None, duplicate: bool = False):
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

    def eval_terms(self, context: dict) -> xr.DataArray:
        """Sum all the terms of the potential, evaluated in a given context.

        Args:
            context (dict): The evaluation context, containing every name the expressions refer to.

        Returns:
            xr.DataArray: The resulting potential.
        """
        V = 0
        for term in self.terms.values():
            V = V + eval(term, {"__builtins__": {}}, context)
        return V

    ### ====================
    ### Plotting
    ### ====================

    def _resolve_time(
        self,
        t: float | None = None,
        t_coord: tuple[float, float, int] | xr.DataArray | None = None,
    ) -> float | xr.DataArray:
        """Return the time on which to evaluate the functions, from the two ways of specifying it.

        Args:
            t (float, optional): A single time.
            t_coord (Union[tuple[float,float,int], xr.DataArray], optional): A whole time dimension,
            either as a tuple (tmin, tmax, n_points) to create a linear array, or directly as a time
            coordinate.

        Raises:
            ValueError: If neither t nor t_coord is given.

        Returns:
            Union[float, xr.DataArray]: The time.
        """
        if t is not None:
            return t
        if t_coord is None:
            raise ValueError("Either 't' or 't_coord' must be given")
        if isinstance(t_coord, tuple):
            return create_parameter("t", np.linspace(*t_coord))
        return t_coord

    def _default_axes(self, cart_axes: list[int] | None = None) -> list[int]:
        """Return the cartesian axes to plot against, defaulting to the first two (or the only one
        in 1D).

        Args:
            cart_axes (list[int], optional): The axes asked for, if any.
        """
        if cart_axes is not None:
            return cart_axes
        return [0] if self.n_dims == 1 else [0, 1]

    def plot_timefunction(
        self, name: str | list[str], tmin: float, tmax: float, n_t: int = 100
    ) -> tuple[Figure, Axes]:
        """Create an interactive plot showing a time function with its parameters as sliders. Can plot multiple time functions at the same time.
        Functions that also depend on space are sampled at the origin.

        Args:
            name (Union[str, list[str]]): The name (names) of the time function(s) to plot.
            tmin (float): lower bound for the plotting window.
            tmax (float): Upper bound for the plotting window.
            n_t (int, optional): Number of points to compute. Defaults to 100.

        Returns:
            tuple[Figure, Axes]: Figure and axes objects to be plotted.
        """
        t = create_parameter("t", np.linspace(tmin, tmax, n_t))
        origin = [0.0] * self.n_dims

        def evaluate(na: str) -> xr.DataArray:
            # The '+ 0 * t' keeps a time dimension even for a function that is constant in time
            return self.eval_func(na, t, origin) + 0 * t

        if isinstance(name, str):
            return plot_cuts(evaluate(name).squeeze(), "t", groupby=[])

        elif isinstance(name, list):
            ndim = create_parameter("funcs", np.arange(len(name)))
            arr = xr.concat([evaluate(na) for na in name], dim=ndim, coords="minimal")

            fig, ax = plot_cuts(arr.squeeze(), "t", groupby=["funcs"])
            ax.legend(name)
            return fig, ax

        else:
            raise ValueError("'name' must be a list or a string.")


class PotentialT(ParametricPotential):
    """A time-dependent potential, built as a combination of static spatial shapes modulated by
    functions of time. Its 'make_Vt' method returns the fast V(t) evaluation used by the solvers."""

    def __init__(
        self,
        unitvecs: list[list[float]],
        resolution: tuple[int],
        v0: complex | np.generic | xr.DataArray = 100,
        dtype: type[int] | type[float] | type[complex] | type[np.generic] = float,
        endpoint: bool = False,
    ):
        """Initialize a PotentialT object, see the Potential class for the meaning of the arguments.
        The dimension of the described space (1D, 2D or 3D) is set by the number of unit vectors given.
        """
        super().__init__(unitvecs, resolution, v0, dtype, endpoint)

        self.shapes_t = self._initial_shapes()  # The spatial parts of the potential
        self.terms = {"V0": "V0"}  # The time-independent part is a term like any other
        self.context_funcs: dict[str, Callable] = {}  # Callables usable inside the expressions

    def clear(self):
        """Remove all parameter dimensions and features from the potential"""
        super().clear()
        self.shapes_t = self._initial_shapes()
        self.terms = {"V0": "V0"}
        self.context_funcs = {}

    @property
    def timefuncs(self) -> dict[str, dict]:
        """The registry of time functions. Historical name of 'funcs'."""
        return self.funcs

    def _initial_shapes(self) -> dict[str, xr.DataArray]:
        """The shapes always available to the term expressions: the time-independent potential V0,
        and one entry per cartesian axis ('x', 'y', 'z', up to the dimensionality of the potential).
        """
        shapes = {"V0": self.V}
        shapes.update(
            {self.coord_names[i]: self.coords[i] for i in range(self.n_dims)}
        )
        return shapes

    @staticmethod
    def fromPotential(pot: Potential) -> "PotentialT":
        """Returns a PotentialT object constructed from a Potential object, with only a time independant part.

        Args:
            pot (Potential): The potential to convert
        """

        potT = PotentialT(pot.a, pot.resolution, pot.v0, pot.dtype, pot.endpoint)
        potT.V = pot.V
        potT.update_V0()
        return potT

    def update_V0(self):
        """Update the time independant term of the potential"""
        self.param_coords.update(
            {
                dim: self.V.coords[dim]
                for dim in self.V.dims
                if dim not in self.spatial_dims
            }
        )
        self.shapes_t.update({"V0": self.V})

    ### ====================
    ### Shapes
    ### ====================

    def add_shape(self, name: str, shape: xr.DataArray):
        """Add a time-dependant data array to the object.

        Args:
            name (str): Name of the data.
            shape (xr.DataArray): Time dependant part of the potential, can have multiple dimensions,
            and needs at least to have all the spatial dimensions ('a1', 'a2', ...).

        Raises:
            ValueError: If the shape is missing one of the spatial dimensions.
        """

        missing = [dim for dim in self.spatial_dims if dim not in shape.dims]
        if missing:
            raise ValueError(f"The shape given does not have the dimension(s) {missing}")

        self.param_coords.update(
            {
                dim: shape.coords[dim]
                for dim in shape.dims
                if dim not in self.spatial_dims
            }
        )
        self.shapes_t.update({name: shape})

    def add_context_func(self, name: str, func: Callable):
        """Add an arbitrary function to the evaluation context, to be called by the term expressions.

        Args:
            name (str): Name of the function.
            func (Callable): A function with any numbers of parameters. As the evaluation context
            only supports the base variables "t" and the cartesian coordinates, every argument
            passed when adding terms must be one of those or a function thereof.
        """
        self.context_funcs.update({name: func})

    def _add_mask_shape(
        self,
        name: str,
        mask: xr.DataArray,
        value: float | xr.DataArray,
        inverse: bool,
    ):
        """Register a shape worth 'value' wherever 'mask' is True (or False, if inverse), and 0
        elsewhere. Shared by circle_t, rectangle_t and ellipse_t.

        Args:
            name (str): The name of this time-dependant feature.
            mask (xr.DataArray): A boolean array, True inside the region to fill.
            value (Union[float,xr.DataArray]): The value to set inside the region.
            inverse (bool): Whether to fill the potential inside (False) or outside (True) the region.
        """
        v1, v2 = (0, value) if inverse else (value, 0)
        self.add_shape(name, xr.where(mask, v1, v2))

    def circle_t(
        self,
        name: str,
        center: tuple[float | xr.DataArray],
        radius: float | xr.DataArray,
        inverse: bool = False,
        value: float | xr.DataArray = 1,
    ):
        """Add a shape worth 'value' in a n-sphere. Support coordinates attribution for all parameters.

        Args:
            name (str): The name of this time-dependant feature.
            center (tuple[Union[float,xr.DataArray]]): The center of the n-sphere in the cartesian basis.
            radius (Union[float,xr.DataArray]): The radius of the n-sphere.
            inverse (bool, optional): Whether to fill the potential inside (False) or outside the n-sphere (True).
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the n-sphere. Defaults to 1.
        """
        r = 0
        for i in range(self.n_dims):
            r = r + (self.coords[i] - center[i]) ** 2
        r = r**0.5

        self._add_mask_shape(name, r < radius, value, inverse)

    def rectangle_t(
        self,
        name: str,
        center: tuple[float | xr.DataArray],
        dims: tuple[float | xr.DataArray],
        rotation: tuple[float | xr.DataArray] = (0, 0),
        inverse: bool = False,
        value: float | xr.DataArray = 1,
    ):
        """Add a shape worth 'value' in a rectangle (1D: segment, 2D: rectangle, 3D: rectangular prism).
        Support coordinates attribution for all parameters.

        Args:
            name (str): The name of this time-dependant feature.
            center (tuple[Union[float,xr.DataArray]]): The center of the rectangle in the cartesian basis.
            dims (tuple[Union[float,xr.DataArray]]): The side lengths, one per dimension.
            rotation (tuple[Union[float,xr.DataArray]], optional): A rotation (in radians) of the rectangle,
            as a tuple of angles. See Potential.rotate_center for details. Defaults to (0, 0).
            inverse (bool, optional): Whether to fill the potential inside (False) or outside the rectangle (True).
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 1.
        """
        coords = self.rotate_center(center, rotation)

        # Inside the rectangle iff within bounds on every axis
        mask = abs(coords[0]) < dims[0] / 2
        for i in range(1, self.n_dims):
            mask = mask & (abs(coords[i]) < dims[i] / 2)

        self._add_mask_shape(name, mask, value, inverse)

    def ellipse_t(
        self,
        name: str,
        center: tuple[float | xr.DataArray],
        dims: tuple[float | xr.DataArray],
        rotation: tuple[float | xr.DataArray] = (0, 0),
        inverse: bool = False,
        value: float | xr.DataArray = 1,
    ):
        """Add a shape worth 'value' in an ellipse (1D: segment, 2D: ellipse, 3D: ellipsoid).
        Support coordinates attribution for all parameters.

        Args:
            name (str): The name of this time-dependant feature.
            center (tuple[Union[float,xr.DataArray]]): The center of the ellipse in the cartesian basis.
            dims (tuple[Union[float,xr.DataArray]]): The semi-axes, one per dimension.
            rotation (tuple[Union[float,xr.DataArray]], optional): A rotation (in radians) of the ellipse axes,
            as a tuple of angles. See Potential.rotate_center for details. Defaults to (0, 0).
            inverse (bool, optional): Whether to fill the potential inside (False) or outside the ellipse (True).
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the ellipse. Defaults to 1.
        """
        coords = self.rotate_center(center, rotation)

        r = 0
        for i in range(self.n_dims):
            r = r + (coords[i] / dims[i]) ** 2
        r = r**0.5

        self._add_mask_shape(name, r < 1, value, inverse)

    def modulate(self, expression: str):
        """A high level function to modulate the time-independant potential V0 by a time function.

        Args:
            expression (str): The modulation expression as a string to be evaluated
        """
        self.terms["V0"] = f"V0 * {expression}"

    ### ====================
    ### Evaluation
    ### ====================

    def to_potential(
        self,
        t: float | None = None,
        t_coord: tuple[float, float, int] | xr.DataArray | None = None,
    ) -> Potential:
        """Return a Potential object evaluated at a specified time t or with a dimension t. Useful for plotting functions.

        Args:
            t (float, optional): If not None, then the potential returned is evaluated at the time t.
            t_coord (Union[tuple[float,float,int], xr.DataArray], optional): Specify the time dimension to add, can either be a tuple (tmin, tmax, n_points)
            to create a linear array, or directly a time coordinate xarray.

        Returns:
            Potential: A Potential object.
        """
        time = self._resolve_time(t, t_coord)
        self.update_V0()

        context = {"np": np, "xr": xr, **self.context_funcs, **self.shapes_t}
        for name in self.funcs:
            context.update({name: self.eval_func(name, time)})

        pot = Potential(
            self.a, self.resolution, v0=0, dtype=self.dtype, endpoint=self.endpoint
        )
        pot.set(self.eval_terms(context))  # 'set' broadcasts the terms over the whole grid
        return pot

    def plot_t(
        self,
        tmin: float,
        tmax: float,
        n_t: int = 100,
        cart_axes: list[int] | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Creates an interactive plot of the potential, with the time and all the parameters as sliders.
        Must be used in an interactive python session, preferably a notebook.

        Args:
            tmin (float): lower bound for the time window.
            tmax (float): upper bound for the time window.
            n_t (int, optional): Number of times to compute. Defaults to 100.
            cart_axes (list[int], optional): The cartesian axes indexes to plot against, with 0 = "x",
            1 = "y" and 2 = "z". Defaults to the first two axes, or the only one in 1D.
            **kwargs: passed on to the matplotlib function used (plt.plot or plt.pcolormesh).
        """
        pot = self.to_potential(t_coord=(tmin, tmax, n_t))
        return pot.plot(cart_axes=self._default_axes(cart_axes), **kwargs)

    def make_Vt(self, selection: dict[str, float]) -> Callable:
        """The main point of this class. Given a selection of value for each parameter dimension,
        return a function who takes as input the time and returns the potential landscape at this time.
        This function is very fast and intended for evaluation during time-dependant simulations.

        Args:
            selection (dict[str,float]): The selection of parameters value.

        Returns:
            Callable: The function V(t) given the selection of parameter values.
        """
        self.update_V0()

        funcs_sel = {name: self.create_func(name, selection) for name in self.funcs}

        shapes_sel = {
            name: shape.sel({dim: selection[dim] for dim in selection if dim in shape.dims})
            for name, shape in self.shapes_t.items()
        }

        context = {"np": np, "xr": xr, **self.context_funcs, **shapes_sel}

        def Vt(t: float):
            for name, func in funcs_sel.items():
                context.update({name: func(t)})

            return self.eval_terms(context)

        return Vt


class AnalyticPotential(ParametricPotential):
    """A time-dependent potential defined only by analytical functions of time and space, without
    any static grid of its own. This class is designed to work with the Rescaling GP solver, which
    needs to evaluate the potential on a grid that changes at every time step."""

    def __init__(
        self,
        unitvecs: list[list[float]],
        resolution: tuple[int],
        dtype: type[int] | type[float] | type[complex] | type[np.generic] = float,
        endpoint: bool = False,
    ):
        """Initialize an AnalyticPotential object. Contrarily to the other potential classes, the
        grid described by the unit vectors is only the default one: the potential can be evaluated
        on any set of coordinates.

        Args:
            unitvecs (list[list[float]]): Unit vectors for the invariant grid, one per dimension (1, 2 or 3 of them).
            resolution (tuple[int]): Grid resolution along each unit vector.
            dtype (Union[Type[int],Type[float],Type[complex],Type[np.generic]], optional): The Potential type. Defaults to float.
            endpoint (bool, optional): Whether to add the endpoint to the a1, a2, ... linspaces. Defaults to False.
        """
        super().__init__(unitvecs, resolution, 0, dtype, endpoint)

    @staticmethod
    def fromPotential(pot: Potential) -> "AnalyticPotential":
        """Returns an AnalyticPotential object constructed from a Potential object, with the same
        resolution and simulation box. The landscape of 'pot' is NOT carried over, as an
        AnalyticPotential is defined by its analytical terms only.

        Args:
            pot (Potential): The potential to convert.
        """

        return AnalyticPotential(pot.a, pot.resolution, pot.dtype, pot.endpoint)

    def _wrap_timefunc(self, fc: Callable) -> Callable:
        """The functions of this class are called with the coordinates, which a pure time function
        simply ignores."""
        return lambda t, *coords, **kwargs: fc(t, **kwargs)

    def _call_func(self, func: Callable, t: float | xr.DataArray, coords: list, kwargs: dict):
        """Evaluate a registered function on the time and the coordinates, defaulting to the
        potential's own grid."""
        coords = self.coords if coords is None else coords
        return func(t, *coords, **kwargs)

    def to_potential(
        self,
        t: float | None = None,
        t_coord: tuple[float, float, int] | xr.DataArray | None = None,
        coords: list[xr.DataArray] | None = None,
    ) -> Potential:
        """Return a Potential object evaluated at a specified time t or with a dimension t. Useful for plotting functions.

        Args:
            t (float, optional): If not None, then the potential returned is evaluated at the time t.
            t_coord (Union[tuple[float,float,int], xr.DataArray], optional): Specify the time dimension to add, can either be a tuple (tmin, tmax, n_points)
            to create a linear array, or directly a time coordinate xarray.
            coords (list[xr.DataArray], optional): The parameter dependant coordinates at which to evaluate the potential,
            one array per dimension. If none are given, then the base grid defined by the unit vectors is used.

        Returns:
            Potential: A Potential object.
        """
        time = self._resolve_time(t, t_coord)

        context = {"np": np, "xr": xr}
        for name in self.funcs:
            context.update({name: self.eval_func(name, time, coords)})

        pot = Potential(
            self.a, self.resolution, v0=0, dtype=self.dtype, endpoint=self.endpoint
        )
        pot.set(self.eval_terms(context))  # 'set' broadcasts the terms over the whole grid
        return pot

    def plot_t(
        self,
        tmin: float,
        tmax: float,
        n_t: int = 100,
        cart_axes: list[int] | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Creates an interactive plot of the potential over the base grid, with the time and all the
        parameters as sliders. Must be used in an interactive python session, preferably a notebook.

        Args:
            tmin (float): lower bound for the time window.
            tmax (float): upper bound for the time window.
            n_t (int, optional): Number of times to compute. Defaults to 100.
            cart_axes (list[int], optional): The cartesian axes indexes to plot against, with 0 = "x",
            1 = "y" and 2 = "z". Defaults to the first two axes, or the only one in 1D.
            **kwargs: passed on to the matplotlib function used (plt.plot or plt.pcolormesh).
        """
        pot = self.to_potential(t_coord=(tmin, tmax, n_t))
        return pot.plot(cart_axes=self._default_axes(cart_axes), **kwargs)

    def plot(self, *args, **kwargs) -> tuple[Figure, Axes]:
        """An AnalyticPotential has no static landscape to plot, so 'plot' is an alias of 'plot_t'."""
        return self.plot_t(*args, **kwargs)

    def make_V(self, selection: dict[str, float]) -> Callable:
        """The main point of this class. Given a selection of value for each parameter dimension,
        return a function who takes as input the time and the coordinates, and returns the potential
        landscape there. This function is very fast and intended for evaluation during time-dependant
        simulations.

        Args:
            selection (dict[str,float]): The selection of parameters value.

        Returns:
            Callable: The function V(t, *coords) given the selection of parameter values. Called
            without coordinates, it evaluates the potential over the potential's own grid.
        """
        funcs_sel = {name: self.create_func(name, selection) for name in self.funcs}

        context = {"np": np, "xr": xr}
        grid = np.zeros(self.resolution)  # ensures the result covers the whole grid

        def V(t: float, *coords):
            coords = coords if coords else self.coords
            for name, func in funcs_sel.items():
                context.update({name: func(t, *coords)})

            return grid + self.eval_terms(context)

        return V

    def make_Vtxy(self, selection: dict[str, float]) -> Callable:
        """The 2D-flavoured name of 'make_V', kept for the solvers that still call it."""
        return self.make_V(selection)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t_pulse = create_parameter("t_pulse", np.linspace(1, 3, 3))
    sigma_pulse = create_parameter("sigma_pulse", np.linspace(1, 2, 2))

    # --- A time-dependent potential, here in 3D ---
    foo = PotentialT([[2, 0, 0], [0, 2, 0], [0, 0, 2]], (32, 32, 32), v0=0)

    foo.gaussian("pulse", t_pulse, sigma_pulse, norm="peak")
    foo.ellipse_t("blob", (0, 0, 0), (0.5, 0.3, 0.2), value=1)
    foo.add_term("blob * pulse")

    Vt = foo.make_Vt({"t_pulse": 2, "sigma_pulse": 1})
    print(foo, "\n V(t=2) max:", float(Vt(2).max()))

    # --- The same profile, this time as an analytical potential ---
    bar = AnalyticPotential([[2, 0, 0], [0, 2, 0], [0, 0, 2]], (32, 32, 32))

    bar.gaussian("pulse", t_pulse, sigma_pulse, norm="peak")
    bar.add_function("harm", lambda t, x, y, z, sigma: (x**2 + y**2 + z**2) * sigma / 2,
                     parameters={"sigma": sigma_pulse})
    bar.add_term("harm * pulse")

    V = bar.make_V({"t_pulse": 2, "sigma_pulse": 1})
    print(bar, "\n V(t=2) max:", float(V(2).max()))

    bar.plot_t(0, 2)
    plt.show()
