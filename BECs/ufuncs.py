import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian(x, y, x0, y0, sigmax, sigmay, amp, theta=0):
    """Create a gaussian profile, every parameter can be set up using scalar or arrays.

    Args:
        x (Any): x-coordinate
        x0 (Any): x coordinate of the center point
        sigmax (Any): half width of the function along x
        amp (Any): Amplitude of the distribution at its center.

    Returns:
        Any: Height at x
    """
    return amp * np.exp(-((x - x0) ** 2 / (2 * sigmax**2)))


def gaussian2D(x, y, x0, y0, sigmax, sigmay, amp, theta=0):
    """Create a 2D gaussian profile, every parameter can be set up using scalar or arrays.

    Args:
        x (Any): x-coordinate
        y (Any): y-coordinate
        x0 (Any): x coordinate of the center point
        y0 (Any): y coordinate of the center point
        sigmax (Any): half width of the function along x
        sigmay (Any): half width of the function along y
        amp (Any): Amplitude of the distribution at its center.
        theta (Any): Rotation of the x and y axes in radians. default to 0

    Returns:
        Any: Height at x, y
    """
    lx = 2 * sigmax**2
    ly = 2 * sigmay**2
    a = np.cos(theta) ** 2 / lx + np.sin(theta) ** 2 / ly
    b = np.sin(theta) * np.cos(theta) * (-1 / lx + 1 / ly)
    c = np.sin(theta) ** 2 / lx + np.cos(theta) ** 2 / ly

    return amp * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )


def harmonic2D(x, y, x0, y0, omega_x, omega_y, theta=0):
    """Create an arbitrary 2D harmonic trap. Every parameter can be set up using scalar or arrays.

    Args:
        x (Any): x-coordinate.
        y (Any): y-coordinate.
        x0 (Any): x coordinate of the bottom point.
        y0 (Any): y coordinate of the bottom point.
        omega_x (Any): trap frequency along x.
        omega_y (Any): trap frequency along y.
        theta (Any): Rotation of the x and y axes in radians. default to 0

    Returns:
        Any: Height at x, y
    """
    a = np.cos(theta) ** 2 * omega_x / 2 + np.sin(theta) ** 2 * omega_y / 2
    b = np.sin(theta) * np.cos(theta) * (omega_x / 2 + omega_y / 2)
    c = np.sin(theta) ** 2 * omega_x / 2 + np.cos(theta) ** 2 * omega_y / 2

    return a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2


def ramp(
    t: xr.DataArray,
    vi: float,
    vf: float,
    t_start: float,
    t_stop: float,
    smooth: float = 0,
) -> xr.DataArray:
    """Create a smooth ramp function.

    Args:
        t (xr.DataArray): time coordinate.
        vi (float): Initial value of the ramp (for t<t_start)
        vf (float): Final value of the ramp (for t>t_stop)
        t_start (float): Start time of the ramp
        t_stop (float): End time of the ramp.
        smooth (float, optional): A smoothing of the ramp, in index coordinates. Defaults to 0.

    Returns:
        xr.DataArray: Value of teh ramp sampled on t
    """
    ramp = xr.where(t < t_start, vi, 0)
    ramp = xr.where(t > t_stop, vf, ramp)
    ramp = xr.where(
        (t_start <= t) * (t <= t_stop),
        vi + (vf - vi) * (t - t_start) / (t_stop - t_start),
        ramp,
    )

    ramp.data = gaussian_filter(ramp, sigma=(smooth))
    return ramp
