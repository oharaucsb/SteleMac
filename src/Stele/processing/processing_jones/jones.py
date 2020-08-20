from __future__ import division
import numpy as np


p0 = [0.5, 2, 90., 0.5]


class AngleWrapper(object):
    """
    Class which will force angles to be within the specified bounds
    """
    def __init__(self, mn, mx):
        self._min = mn
        self._max = mx

    def __contains__(self, item):
        return self._min < item < self._max

    def wrap(self, datum):
        bnd = (self._max - self._min)
        datum = np.array(datum)
        datum[datum > self._max] = abs(datum[datum > self._max]) - bnd
        datum[datum < self._min] = bnd - abs(datum[datum < self._min])
        return datum


def cos(x):
    return np.cos(x * np.pi/180.)


def sin(x):
    return np.sin(x * np.pi/180.)


exp = np.exp
pi = np.pi

pideg = 180./pi
degpi = pi/180.


def printMat(m):
    print('[')
    if m.ndim == 1:
        print(
            "\t" + ', '.join(["{:.3f} exp({:.3f}i)".format(
                np.abs(ii), np.angle(ii, deg=True)) for ii in m]))
    else:
        for r in m:
            print(
                "\t[" + ', '.join(["{:.3f} exp({:.3f}i)]".format(
                    np.abs(ii), np.angle(ii, deg=True)) for ii in r]) + "]")
    print(']')


def make_birefringent(theta, eta=pi, phi=0):
    """

    :param theta: Angle fast axis makes with horizontal
    :param eta:  retardance
    :param phi: "circularity"?
    :return: Jones matrix for the given parameters
    """
    a = exp(1j * eta/2)*cos(theta)**2 + exp(-1j * eta/2)*sin(theta)**2
    b = (exp(1j * eta/2) - exp(-1j * eta/2))*exp(-1j*phi)*cos(theta)*sin(theta)
    c = (exp(1j * eta/2) - exp(-1j * eta/2))*exp(1j*phi)*cos(theta)*sin(theta)
    d = exp(1j * eta/2)*sin(theta)**2 + exp(-1j * eta/2)*cos(theta)**2

    retMat = np.array([[a, b], [c, d]])

    # need to remove issues from machine precision
    retMat.real[abs(retMat.real) < 1e-4] = 0.0
    retMat.imag[abs(retMat.imag) < 1e-4] = 0.0

    return retMat


def make_rotation(theta):
    """
    makes a rotation matrix
    Note: HWP is a mirror matrix, not a rotation!
    :param theta:
    :return:
    """
    a = cos(theta)
    b = -sin(theta)
    c = sin(theta)
    d = cos(theta)

    retMat = np.array([[a, b], [c, d]])

    return retMat


def dot(m, v):
    """
    will handle matrix multiplication of matrix/vector
    Was designed to handle it when one of them has an extra
    dimension corresponding to some time-like coordinate (some independent
    parameter which varies).

    I started making this when I was having trouble with einsum, but
    figured those issues out before finalizing this code.

    If this stuff is ever needed for some reason and einsum won't work,
    make sure to test this, I'm not 100% it works like it's supposed to.
    :param m: Matrix
    :param v: Vector
    :type m: np.ndarray
    :type v: np.ndarray
    :return:
    """
    raise NotImplementedError
    if m.ndim == 3 and v.ndim == 3:
        pass
    else:
        print("input m", m.shape)
        print("input v", v.shape)
    if m.ndim == 3 and v.ndim == 1:
        newv = v[:, None, None] * np.ones(m.shape[2])[None, None, :]
        return dot(m, newv)
    if m.ndim == 2 and v.ndim == 1:
        newm = m[:, :, None]
        newv = v[:, None, None] * np.ones(newm.shape[2])[None, None, :]
        return dot(newm, newv)[:, 0, 0]
    if m.ndim == 3 and v.ndim == 2:
        newv = v[:, :, None] * np.ones(m.shape[2])[None, None, :]
        return dot(m, newv)
    elif m.ndim == 2 and v.ndim == 2:
        return dot(m[:, :, None], v[:, :, None])[:, :, 0]
    elif m.ndim == 2 and v.ndim == 3:
        newm = m[:, :, None] * np.ones(v.shape[2])[None, None, :]
        return dot(newm, v)
    elif m.ndim == 3 and v.ndim == 3:
        retmat = []
        for idx in range(m.shape[2]):
            retmat.append(np.dot(m[:, :, idx], v[:, :, idx]))
        print("return shape", np.array(retmat).shape)
        return np.array(retmat).transpose([1, 2, 0])
