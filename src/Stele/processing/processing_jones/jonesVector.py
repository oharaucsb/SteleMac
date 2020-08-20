from __future__ import division
import numpy as np
from .processsing_jones import jones
import numpy.pi as pi


pideg = 180./pi
degpi = pi/180.


class JonesVector(object):
    def __init__(
            self, Ex=1, Ey=0, phi=None, delta=None, alpha=None, gamma=None):
        """
        A class containing a jones vector which can be manipulated (sent
        through some birefrengent material). Several initializers are possible.

        self.vec is a 1D, len 2 complex vector where the 0th index is the
        magnitude of Ex and 1st is Ey. All phase is kept on Ey, and vector is
        kept normalized. That is, vec = [cos(phi), exp(i delta)sin(phi)].

        Specifying Ex and Ey will set the inputs directly.

        Specifying phi and delta will define the vector as listed above

        Specifying alpha and gamma will define elliptical light whose semimajor
        axis makes an angle alpha with horizontal, and whose semimajor/minor
        axis define a right triangle with angle gamma (tan gamma = min/maj)

        all angles should be in degrees

        """

        if isinstance(Ex, JonesVector):
            self.vec = Ex.vec.copy()
            return
        elif None not in [phi, delta]:
            phi = np.array(phi)
            delta = np.array(delta)
            Ex = jones.cos(phi)
            Ey = jones.sin(phi) * np.exp(1j*delta * np.pi/180.)
        elif alpha is not None and gamma is not None:
            """
            I've found that the proper way to construct a jones vector from
            elliptical coordinates is to create elliptical oriented along the
            horizontal, where Ex and Ey must be 90deg out of phase, and then
            rotate the vector.
            """
            alpha = np.array(alpha)
            gamma = np.array(gamma)
            v = np.array([jones.cos(gamma), 1j*jones.sin(gamma)])
            r = jones.make_rotation(alpha)
            # pass single alpha and gamma
            if alpha.size == 1 and gamma.size == 1:
                v = np.einsum("ij,j->i", r, v)
                Ex = np.abs(v[0])
                Ey = np.abs(v[1]) * np.exp(1j * np.diff(np.angle(v))[0])
            # pass single alpha, multiple gamma
            elif alpha.size == 1:
                v = np.einsum("ij,jk->ik", r, v)
                Ex = np.abs(v[0])
                Ey = np.abs(v[1]) * np.exp(1j * np.diff(
                    np.angle(v), axis=0)[0])
            # pass single gamma, multiple alpha
            elif gamma.size == 1:
                v = np.einsum("ijk,j->ik", r, v)
                Ex = np.abs(v[0])
                Ey = np.abs(v[1]) * np.exp(1j * np.diff(
                    np.angle(v), axis=0)[0])
            # pass multiple alpha/gamma
            else:
                v = np.einsum("ijk,jk->ik", r, v)
                Ex = np.abs(v[0, :])
                Ey = np.abs(v[1, :]) * np.exp(1j * np.diff(
                    np.angle(v), axis=0)[0, :])

        mag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
        Ex /= mag
        Ey /= mag

        self.vec = np.array([Ex, Ey])

    def __repr__(self):
        st = '['
        if self.vec.ndim == 1:
            st += "\t" + ', '.join(["{:.3f} exp({:.3f}i)".format(
                np.abs(ii), np.angle(ii, deg=True)) for ii in self.vec])
        else:
            for r in self.vec:
                st += "\t[" + ', '.join(["{:.3f} exp({:.3f}i)]".format(
                    np.abs(ii), np.angle(ii, deg=True)) for ii in r]) + "]\n"
        st += ']'

        return st

    @property
    def x(self):
        return self.horizontal_projection()[0]

    @property
    def y(self):
        return self.vertical_projection()[1]

    @property
    def delta(self):
        # v = self.vec[:,None,None]
        v = self.vec
        ang = np.angle(v, deg=True)
        return np.squeeze(ang[1]-ang[0])

    # TODO: determine what pideg and degpi are meant to be and assign them
    @property
    def phi(self):
        # v = self.vec[:,None,None]
        v = self.vec
        mag = np.abs(v)
        return np.squeeze(np.arctan2(mag[1], mag[0]))*pideg

    @property
    def alpha(self):
        phi = self.phi * degpi
        delta = self.delta * degpi

        return np.squeeze(np.arctan2(
                np.abs(2*np.tan(phi))*np.cos(delta), (1-np.tan(phi)**2)
            )
            )/2.*pideg

        return np.squeeze(np.arctan2(
                np.abs(2*np.tan(phi)/(1-np.tan(phi)**2))*np.cos(delta), 1.
            )
        )/2.*pideg


    @property
    def gamma(self):
        phi = self.phi * degpi
        delta = self.delta * degpi

        return np.squeeze(np.arcsin(
            2*np.tan(phi)/(1+np.tan(phi)**2) * np.sin(delta)))/2.*pideg

    @property
    def oldalpha(self):
        # v = self.vec[:,None,None]
        v = self.vec
        ang = np.angle(v)
        mag = np.abs(v)
        return np.squeeze(np.arctan2(
            2*mag[0]*mag[1]*np.cos(ang[1]-ang[0]),
            (mag[0]**2-mag[1]**2))/2*180/3.14159)

    @property
    def oldgamma(self):
        # v = self.vec[:,None,None]
        v = self.vec
        mag = np.abs(v)

        return np.squeeze(np.arcsin(
            2*mag[0]*mag[1]*np.sin(self.delta*np.pi/180)
            / (mag[0]**2+mag[1]**2))/2*180/3.14159)

    def apply_hwp(self, theta=0):
        """
        Apply a hwp rotation to the jones vector with a fast axis
        aligned at an angle theta with respect to horizontal.
        :param theta: Angle of HWP wrt horizontal (in deg)
        :return:
        """
        self.apply_retarder(theta, eta=np.pi)

    def apply_qwp(self, theta=0):
        """
        Apply a qwp rotation to the jones vector with a fast axis
        aligned at an angle theta with respect to horizontal.
        :param theta: Angle of qWP wrt horizontal (in deg)
        :return:
        """
        self.apply_retarder(theta, eta=np.pi/2.)

    def apply_retarder(self, theta=0, eta=0):
        """
        Apply an arbitrary retarder to the
        internal vector.


        As such, there's a lot of checks on the number of dimensions of the
        transformation and the vector to intelligently handle what things
        should be contracted.

        :param theta:
        :param eta:
        :return:
        """
        theta = np.array(theta)
        eta = np.array(eta)
        if theta.ndim == eta.ndim == 1:
            theta = theta[:, None]
            if len(theta) == len(eta) != 1:
                msg = "Caution: I don't think things will work if you have" \
                    "theta/eta being the same dimensions"
                raise ValueError(msg)
        transform = jones.make_birefringent(theta, eta)

        ret = self.apply_transformation(transform)

    def vertical_projection(self):
        m = [[0, 0], [0, 1]]
        return self.apply_transformation(m, update_state=False)

    def horizontal_projection(self):
        m = [[1, 0], [0, 0]]
        return self.apply_transformation(m, update_state=False)

    def apply_transformation(self, transform, update_state=True):
        """
        I tried to make this really general. If you pass it an array of
        theta/eta because you want to look at rotating the FA (via theta) or
        different retardences (via eta), it'll do them all here, instead of
        needing to loop over it.  Hopefully this odesn't lead to complications,
        and I hope I have all the indices proper.

        :param transform: Arbitrary 2x2(xNxM) matrix. Could probably be much
            less hardcoded, but it was easier to do this way for now.
        :return:
        """
        transform = np.array(transform)
        if not transform.shape[0] == transform.shape[1] == 2:
            raise ValueError("Transform matrix must have 2x2 on first two"
                             "indices, not {}".format(transform.shape))

        tndim = transform.ndim
        tsh = transform.shape
        vndim = self.vec.ndim
        vsh = self.vec.shape
        if tndim == 2 and vndim == 1:
            einIndices = "ij,j->i"
        elif tndim == 3 and vndim == 1:
            einIndices = "ijm,j->im"
        elif tndim == 4 and vndim == 1:
            einIndices = "ijmn,j->ijm"
        elif tndim == 2 and vndim == 2:
            einIndices = "ij,jk->ik"
        elif tndim == 3 and vndim == 2:
            if tsh[2] == vsh[1]:
                einIndices = "ijm,jm->im"
            else:
                einIndices = "ijm,jk->imk"

        elif tndim == 4 and vndim == 2:
            # also not fully tested
            if tsh[2] == vsh[1]:
                einIndices = "ijkl,jk->ikl"
            elif tsh[3] == vsh[1]:
                einIndices = "ijkl,jl->ikl"
            else:
                msg = "Do not know how to parse transform and vector shapes " \
                      "{} vs {}".format(tsh, vsh)
                raise ValueError(msg)
        elif tndim == 2 and vndim == 3:
            einIndices = "ij,jkl->ikl"
        elif tndim == 3 and vndim == 3:
            # this one hasn't been thouroughly tested
            if tsh[2] == vsh[1]:
                einIndices = "ijm,jmn->imn"
            elif tsh[2] == vsh[2]:
                einIndices = "ijn,jmn->imn"
            else:
                msg = "Do not know how to parse transform and vector shapes " \
                      "{} vs {}".format(tsh, vsh)
                raise ValueError(msg)
        elif tndim == 4 and vndim == 3:
            if tsh[2] == vsh[1] and tsh[3] == vsh[2]:
                einIndices = "ijkl,jkl->ikl"
            elif tsh[2] == vsh[2] and tsh[3] == vsh[1]:
                einIndices = "ijkl,jlk->ikl"
            else:
                msg = "Do not know how to parse transform and vector shapes " \
                      "{} vs {}".format(tsh, vsh)
                raise ValueError(msg)
        else:
            msg = "Do not know how to parse transform and vector shapes " \
                  "{} vs {}".format(tsh, vsh)
            raise ValueError(msg)

        vec = np.einsum(einIndices, transform, self.vec)
        if update_state:
            self.vec = vec
        return vec

    def unwrap_phase(self, input_polarization):
        """
        after putting the vector through a retarder, the phase can get flipped
        (hwp nature)
        :param input_polarization:
        :return:
        """
        pass

    def to_Stokes(self):
        x = np.abs(self.x)**2 - np.abs(self.y)**2
        y = 2 * np.real(self.x*np.conj(self.y))
        z = 2 * np.imag(self.x*np.conj(self.y))
        return np.array([np.ones_like(x), x, y, z])


if __name__ == '__main__':
    from hsganalysis.jones import JonesVector as JV
    a = JV(alpha=[-45, 0, 45, 90], gamma=0)
    a.to_Stokes()
    a = JonesVector(phi=np.arange(0, 90, 5), delta=np.ones(90/5))
    a.gamma
