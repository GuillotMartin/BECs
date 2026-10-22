import builtins
import re
import threading
import warnings
from collections.abc import Callable

import numpy as np
import xarray as xr
from bloch_schrodinger.frames import (
    cart_coord,
    cart_factors,
    lattice_dims,
    moving_dims,
    set_cart_factors,
)
from bloch_schrodinger.plotting import plot_cuts
from bloch_schrodinger.potential import Potential, create_parameter
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def selects_itself(function) -> bool:
    """Whether a function carries parameter dimensions of its own and reduces itself to a point of
    them, through a 'param_coords' attribute and a 'sel' method. See
    'ParametricPotential.add_function' for what such a function is expected to do.
    """
    return (
        callable(function)
        and hasattr(function, "param_coords")
        and callable(getattr(function, "sel", None))
    )


def _on_lattice(value, lattice: list[str], shape: tuple[int]) -> np.ndarray:
    """One frame of a coordinate or of a landscape, as a plain array over the whole lattice.

    Whatever is left of the frame's own dims has size one by then and is dropped, a lattice axis
    the value does not run along is broadcast, and the axes come out in lattice order -- which is
    the order the solvers' plain arrays are in.

    Args:
        value: A DataArray, or anything numpy can broadcast, such as a scalar or a plain grid.
        lattice (list[str]): The lattice dims, in axis order.
        shape (tuple[int]): The lattice's resolution.

    Returns:
        np.ndarray: The value over the lattice, possibly a read-only broadcast view.
    """
    if isinstance(value, xr.DataArray):
        value = value.squeeze([d for d in value.dims if d not in lattice], drop=True)
        value = value.expand_dims([d for d in lattice if d not in value.dims])
        value = value.transpose(*lattice).data
    return np.broadcast_to(np.asarray(value), shape)


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

    A registered function may also depend on parameter dimensions through its own state rather
    than through its keyword arguments -- a precomputed table, an object model of the experiment.
    Such a function declares them in a 'param_coords' attribute, as {dim: coordinate}, and reduces
    itself to a point of them with a 'sel(selection)' method returning a function of the usual
    signature: the same '.sel' idiom as Potential's and xarray's own. See 'add_function'.
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
            # Register the parameter's label *coordinate*, not its values. A derived parameter such
            # as omega = 2*pi/T still carries the label coordinate 'T', and it is the label that
            # 'create_func' selects on further down. Registering 'val' itself made that selection
            # look omega's values up among the T labels, which 'nearest' then silently collapsed
            # onto a single point -- so the whole sweep ran at one parameter value.
            coord = val.coords[val.name] if val.name in val.coords else val
            self.param_coords.update({val.name: coord})
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

        A parameter given as a DataArray is handed to the function as its *value* at the point a
        run sits at. That is all an ordinary function needs, but not one whose dependence on a
        parameter goes through more than a number, such as an object built from swept inputs. Such
        a function can carry its parameter dimensions itself instead:

        - 'function.param_coords', a {dim: coordinate} mapping, lists them. They are registered as
          parameter dimensions of the potential, so the solvers iterate over them.
        - 'function.sel(selection)' reduces it to one point of them, given the label of each, and
          returns a function of the usual signature. The solvers' 'make_Vt'/'make_V' call it once
          per run, so whatever the reduction costs is not paid at every time step.

        Called unreduced, as 'to_potential' and 'plot_t' do, it receives DataArray times and
        coordinates and is expected to keep its own dimensions in the result, exactly as an
        ordinary function keeps those of its DataArray parameters.

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
        dictfunc = {"func": function, "dims": {}, "parameters": {}, "selects": None}

        for key, value in (parameters or {}).items():
            self.add_param(dictfunc, key, value)

        if selects_itself(function):
            self.param_coords.update(function.param_coords)
            dictfunc["selects"] = tuple(function.param_coords)

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
        entry = self.funcs[name]
        kwargs = {**entry["parameters"]}
        for param, values in entry["dims"].items():
            # 'nearest' avoids some rounding errors on the selected values
            kwargs.update(
                {
                    param: values.sel(
                        {dim: selection[dim] for dim in values.dims}, method="nearest"
                    ).data
                }
            )
        func = entry["func"]
        if entry.get("selects") is not None:
            # Only the dims it declared, and only those given: a missing one is the function's to
            # report, since it knows what it needs them for
            func = func.sel({dim: selection[dim] for dim in entry["selects"] if dim in selection})
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

    def is_static(self) -> bool:
        """Whether V(t) is in fact independent of t.

        True when no term expression mentions any registered time function. Most potentials handed
        to the solvers are of this kind -- a plain 'Potential' arrives through 'fromPotential' with
        the single term "V0" -- and knowing it lets the caller evaluate the landscape once for a
        whole run instead of once per step, and cache anything built from it.

        Returns:
            bool: True if the potential does not depend on time.
        """
        names = set(self.funcs)  # every registered time function, the identity 't' included
        for expression in self.terms.values():
            if names & set(re.findall(r"[A-Za-z_]\w*", expression)):
                return False
        return True

    def eval_terms(self, context: dict) -> xr.DataArray:
        """Sum all the terms of the potential, evaluated in a given context.

        The expressions are evaluated with the ordinary builtins in scope. They used to be given an
        empty '__builtins__', which reads as a sandbox but is not one -- the terms are strings the
        user wrote in their own script, and 'context' hands them numpy, so anything reachable by
        eval was already reachable. What the empty mapping did do was break cupy: compiling an
        elementwise kernel goes through an introspection path that looks up '__import__' in the
        calling frame's globals, so a term touching device arrays raised

            KeyError: '__import__'

        from inside cupy's kernel machinery. It surfaced only on a cold kernel cache -- once a
        given operation had been compiled anywhere in the process the cached kernel was used and
        the lookup never happened -- which made it look intermittent and unrelated to the potential.

        Args:
            context (dict): The evaluation context, containing every name the expressions refer to.

        Returns:
            xr.DataArray: The resulting potential.
        """
        V = 0
        for term in self.terms.values():
            V = V + eval(term, {"__builtins__": builtins}, context)
        return V

    ### ====================
    ### Sampling on a field
    ### ====================

    # Whether the landscape is only known on the potential's own grid. Such a potential cannot be
    # asked about any other point, so it can only be sampled on a field that shares that grid.
    _tabulated = True

    def _frame_V(self, selection: dict) -> Callable:
        """The potential at one point of its parameter space, as a function f(t, coords) of the time
        and of one frame's cartesian coordinates, None meaning the potential's own grid.

        This is the hook behind 'sample_on'. The subclasses build it from the very function the
        solvers call, so a frame that is drawn is evaluated exactly as a run evaluated it.

        Args:
            selection (dict): The label of every parameter dimension of the potential.

        Returns:
            Callable: f(t, coords), returning the landscape over the lattice.
        """
        raise NotImplementedError

    def sample_on(self, field: xr.DataArray | None = None) -> xr.DataArray:
        """The landscape as seen by a field, evaluated frame by frame on the field's own grid.

        The result lives exactly where the field does: on its lattice, with its cartesian
        coordinates -- fixed or moving with the parameters, dense or factored (see
        bloch_schrodinger.frames) -- and over the dims the landscape depends on. Those are the
        parameter dims the field shares with the potential, which then follow the field's own
        sliders, plus any the potential sweeps and the field does not. A dim of the field that the
        landscape does not depend on, such as an interaction strength over a fixed grid, is left
        out for xarray to broadcast. A field reduced to a single point of a dimension -- a scalar
        't' or parameter coordinate -- is sampled at that point, and a field without any time at
        t = 0.

        The result is lazy: a dask array with one chunk per frame, each evaluated only when it is
        looked at, through the same 'make_Vt'/'make_V' the solvers call. Drawing a sweep therefore
        costs one frame of potential per frame shown, and shows the landscape the run actually saw.

        Args:
            field (xr.DataArray, optional): The field, typically a solver's result. Without one, a
            potential that does not depend on time returns its landscape, and any other raises,
            having no time to be sampled at. Defaults to None.

        Raises:
            ValueError: Without a field, for a potential that depends on time.
            ValueError: If the field's lattice is not the potential's, or, for a tabulated
            potential, if the field's grid moves or is not the potential's own.

        Returns:
            xr.DataArray: The landscape, named "V".
        """
        import dask.array as dsa  # only needed here, to build the lazy result

        static = self.is_static()
        if field is None:
            if not static:
                raise ValueError(
                    "This potential depends on time, so without a field there is no time to sample "
                    "it at. Pass the field it is drawn with, or use 'to_potential(t=...)'."
                )
            return self.to_potential(t=0).V

        lattice = list(self.spatial_dims)
        resolution = tuple(self.resolution)
        field_lattice = {d: field.sizes[d] for d in lattice_dims(field)}
        if field_lattice != dict(zip(lattice, resolution)):
            raise ValueError(
                f"The field lives on a {field_lattice} lattice and the potential on "
                f"{dict(zip(lattice, resolution))}. The potential is sampled at the field's own "
                "lattice points, so the two have to match."
            )

        # Where the field sits in space, if it says so; otherwise it is taken to sit on the
        # potential's own grid
        names = list(self.coord_names[: self.n_dims])
        factors = cart_factors(field)
        own_grid = not all(name in field.coords or name in factors for name in names)
        moving = [] if own_grid else [d for n in names for d in moving_dims(field, n)]

        if self._tabulated and not own_grid:
            if moving:
                raise ValueError(
                    f"The field's grid moves with {sorted(set(moving))}, but this potential is only "
                    "known on its own fixed grid. Use an AnalyticPotential to follow a moving grid."
                )
            for name, own in zip(names, self.coords):
                if not np.allclose(
                    _on_lattice(cart_coord(field, name), lattice, resolution),
                    _on_lattice(own, lattice, resolution),
                ):
                    raise ValueError(
                        f"The field's '{name}' coordinate is not the potential's own, and this "
                        "potential is only known on its own grid."
                    )

        # The frames: the field's dims the landscape depends on, in the field's order, then the
        # dims only the potential sweeps
        param_dims = list(self.param_coords)
        wanted = {*param_dims, *moving, *([] if static else ["t"])}
        frame_dims = [d for d in field.dims if d in wanted]
        frame_dims += [d for d in param_dims if d not in field.coords]

        coords = {}
        for d in frame_dims:
            source = field[d] if d in field.dims else self.param_coords[d]
            coords[d] = (d, np.asarray(source.values))
        for d in lattice:
            coords[d] = (d, np.asarray(field[d].values))
        for d in [*param_dims, "t"]:
            if d in field.coords and d not in field.dims:
                coords[d] = ((), field.coords[d].values)
        if own_grid:
            for name, own in zip(names, self.coords):
                coords[name] = (lattice, np.array(_on_lattice(own, lattice, resolution)))
        else:
            # Copied as the field stores them, factored ones included: a dense moving frame is the
            # size of the field, and a factored one is rebuilt per frame only
            for name in names:
                for part in (name,) if name in field.coords else factors[name]:
                    coords[part] = (field.coords[part].dims, field.coords[part].data)

        dims = [*frame_dims, *lattice]
        shape = [len(coords[d][1]) for d in frame_dims] + list(resolution)
        template = xr.DataArray(
            dsa.zeros(shape, chunks=[1] * len(frame_dims) + list(resolution)),
            dims=dims,
            coords=coords,
            name="V",
        )
        if not own_grid:
            template = set_cart_factors(
                template, {n: factors[n] for n in names if n not in field.coords}
            )

        cache: dict[tuple, Callable] = {}
        # Held for the whole evaluation of a frame, not just around the cache: the functions the
        # solvers call keep one evaluation context per run, which is right for a run stepping
        # through time but means two frames of one run must not be evaluated at the same time --
        # which dask's threaded scheduler would otherwise do whenever several chunks are computed.
        lock = threading.Lock()

        def frame(block: xr.DataArray) -> np.ndarray:
            """The landscape over one frame, as a plain array of the block's shape."""
            point = {d: block.coords[d].item() for d in param_dims}
            t = block.coords["t"].item() if "t" in block.coords else 0.0
            positions = (
                None
                if own_grid
                else [_on_lattice(cart_coord(block, n), lattice, resolution) for n in names]
            )
            key = tuple(point.values())
            with lock:
                if key not in cache:
                    cache[key] = self._frame_V(point)
                values = _on_lattice(cache[key](t, positions), lattice, resolution)
            return np.array(values).reshape(block.shape)

        if not frame_dims:
            return template.copy(data=frame(template))

        # One frame evaluated up front, which is also what fixes the dtype: a lossy layer makes
        # the landscape complex, whatever the potential was declared as
        dtype = frame(template.isel({d: [0] for d in frame_dims})).dtype
        template = template.astype(dtype)

        def evaluate(block: xr.DataArray) -> xr.DataArray:
            return block.copy(data=frame(block).astype(dtype, copy=False))

        return xr.map_blocks(evaluate, template, template=template)

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
        resolution: int|tuple[int] = None,
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
        return pot.plot(cart_axes=self._default_axes(cart_axes), resolution=resolution, **kwargs)

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

        if self.is_static():
            # Nothing in the terms depends on t, so the landscape is formed once for the whole run
            # rather than re-evaluated -- and handed back as an ndarray, since every multiply in a
            # solver would otherwise go through xarray's dispatch for no benefit. The 'static'
            # attribute tells the callers they may cache what they build from it; see
            # BECs.spectral.PotentialPhase.
            V_static = np.asarray(self.eval_terms(context))

            def Vt(t: float):
                return V_static

            Vt.static = True
            return Vt

        def Vt(t: float):
            for name, func in funcs_sel.items():
                context.update({name: func(t)})

            return self.eval_terms(context)

        Vt.static = False
        return Vt

    def _frame_V(self, selection: dict) -> Callable:
        """See 'ParametricPotential._frame_V'. A tabulated landscape has no use for coordinates."""
        Vt = self.make_Vt(selection)
        return lambda t, coords: Vt(t)

    def sample_on(self, field: xr.DataArray | None = None) -> xr.DataArray:
        """See 'ParametricPotential.sample_on'. Whatever was put in 'V' since the last update is
        folded in first, so that its parameter dims are among those sampled."""
        self.update_V0()
        return super().sample_on(field)


class AnalyticPotential(ParametricPotential):
    """A time-dependent potential defined only by analytical functions of time and space, without
    any static grid of its own. It can therefore be evaluated on a grid that changes at every
    time step."""

    # Defined everywhere by its functions, so it can be sampled on any grid, moving ones included
    _tabulated = False

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
        resolution: int|tuple[int] = None,
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
        return pot.plot(cart_axes=self._default_axes(cart_axes), resolution=resolution, **kwargs)

    def plot(self, *args, **kwargs) -> tuple[Figure, Axes]:
        """An AnalyticPotential has no static landscape to plot, so 'plot' is an alias of 'plot_t'."""
        return self.plot_t(*args, **kwargs)

    def make_V(self, selection: dict[str, float], xp=np) -> Callable:
        """The main point of this class. Given a selection of value for each parameter dimension,
        return a function who takes as input the time and the coordinates, and returns the potential
        landscape there. This function is very fast and intended for evaluation during time-dependant
        simulations.

        Args:
            selection (dict[str,float]): The selection of parameters value.
            xp (module, optional): The array module the potential should be built with, numpy or
            cupy. Two things in here would otherwise pin the result to the host: the zero grid that
            fixes the result's shape, and the "np" the term expressions are evaluated against --
            a user term reading 'np.cos(x)' has to reach cupy's cosine when x lives on the device.
            Adding a host array to a device one raises rather than transferring, by cupy's design,
            so this cannot be left to broadcasting. Defaults to numpy.

        Returns:
            Callable: The function V(t, *coords) given the selection of parameter values. Called
            without coordinates, it evaluates the potential over the potential's own grid, handed
            to the registered functions as plain arrays of 'xp'.
        """
        funcs_sel = {name: self.create_func(name, selection) for name in self.funcs}

        context = {"np": xp, "xr": xr}
        # Broadcast rather than allocated: this only exists to force the result to cover the whole
        # grid, and a full array of zeros is a grid-sized allocation and a grid-sized add on every
        # call to do it.
        grid = xp.zeros((1,) * len(self.resolution))

        # The potential's own grid, for the callers that pass no coordinates -- the fixed-grid SSFM
        # does exactly that at every step. 'self.coords' are DataArrays on the host, so handing
        # them over as they are built every function's share of the landscape on the host even
        # when the field runs on the device, and the device zero grid above then refused to be
        # added to it. They are moved across once here instead, and stripped of their labels:
        # plain arrays are also what a solver evaluating the potential on a grid of its own passes,
        # so the registered functions see the same kind of input whichever solver calls them.
        default_coords = [xp.asarray(np.asarray(c.data)) for c in self.coords]

        def V(t: float, *coords):
            coords = coords if coords else default_coords
            for name, func in funcs_sel.items():
                context.update({name: func(t, *coords)})

            return xp.broadcast_to(grid + self.eval_terms(context), self.resolution)

        return V

    def _frame_V(self, selection: dict) -> Callable:
        """See 'ParametricPotential._frame_V'."""
        V = self.make_V(selection)
        return lambda t, coords: V(t) if coords is None else V(t, *coords)

    def make_Vtxy(self, selection: dict[str, float], xp=np) -> Callable:
        """The 2D-flavoured name of 'make_V', kept for the solvers that still call it."""
        return self.make_V(selection, xp=xp)


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
