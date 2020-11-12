import numpy as np
import scipy.interpolate as spi


# Returns the indices where Trues reside
def my_lambda(x):
    """
    Returns the indices where Trues reside
    """
    return x.nonzero()[0]


def gauss(x, *p):
    """
    Gaussian fit function.

    :param x: The independent variable
    :type x: np.array, or int or float
    :param p: [mean, area, width, y offset] to be unpacked
    :type p: list of floats or ints
    :return: Depends on x, returns another np.array or float or int
    :rtype: type(x)
    """
    mu, A, sigma, y0 = p
    return (A / sigma) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + y0


def handle_nans(y_vals):
    """
    This function removes nans and replaces them with linearly interpolated
    values.  It requires that the array maps from equally spaced x-values.
    Taken from Stack Overflow: "Interpolate NaN values in a numpy array"
    """
    nan_idx = np.isnan(y_vals)
    y_vals[nan_idx] = np.interp(
        my_lambda(nan_idx), my_lambda(~nan_idx), y_vals[~nan_idx])
    return y_vals


def fourier_prep(x_vals, y_vals, num=None):
    """
    This function will take a Nx2 array with unevenly spaced x-values and make
    them evenly spaced for use in fft-related things.

    And remove nans!
    """
    y_vals = handle_nans(y_vals)

    # for some reason kind='quadratic' doesn't work? returns all nans
    spline = spi.interp1d(x_vals, y_vals,
                          kind='linear')
    if num is None:
        num = len(x_vals)
    even_x = np.linspace(x_vals[0], x_vals[-1], num=num)
    even_y = spline(even_x)

    # even_y = handle_nans(even_y)
    return even_x, even_y
