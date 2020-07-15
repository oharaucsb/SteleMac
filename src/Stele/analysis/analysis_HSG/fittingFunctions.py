import numpy as np


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

def lingauss(x, *p):
    """
    Gaussian fit function with a linear offset

    :param x: The independent variable
    :type x: np.array, or int or float
    :param p: [mean, area, width, constant offset of background, slope of background] to be unpacked
    :type p: list of floats or ints
    :return: Depends on x, returns another np.array or float or int
    :rtype: type(x)
    """
    mu, A, sigma, y0, m = p
    return (A / sigma) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + y0 + m * x

def lorentzian(x, *p):
    """
    Lorentzian fit with constant offset

    :param x: The independent variable
    :type x: np.array, or int or float
    :param p: [mean, area, width, constant offset of background, slope of background] to be unpacked
    :type p: list of floats or ints
    :return: Depends on x, returns another np.array or float or int
    :rtype: type(x)
    """
    mu, A, gamma, y0 = p
    return (A / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2)) + y0

def background(x, *p):
    """
    Arbitrary pink-noise model background data for absorbance FFT
    for the intention of replacing a peak in the FFT
    with the background

    :param x: The independent variable
    :type x: np.array, or int or float
    :param p: [proportionality factor, exponent of power law]
    :type p: list of floats or ints
    :return: Depends on x
    :rtype: type(x)
    """
    a, b = p
    return a * (1 / x) ** b

def gaussWithBackground(x, *p):
    """
    Gaussian with pink-noise background function

    :param x: independent variable
    :type x: np.array, or int or float
    :param p: [mean, area, width, constant background, proportionality of power law, exponent of power law]
    :type p: list of floats or ints
    :return: Depends on x
    :rtype: type(x)
    """
    pGauss = p[:4]
    a, b = p[4:]
    return gauss(x, *pGauss) + background(x, a, b)
