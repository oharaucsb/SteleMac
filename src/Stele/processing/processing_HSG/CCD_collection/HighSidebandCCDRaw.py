import numpy as np
import CCD_collection.HighSidebandCCD as HSCCD

np.set_printoptions(linewidth=500)


class HighSidebandCCDRaw(HSCCD.HighSidebandCCD):
    """
    This class is meant for passing in an image file
        (currently supports a 2x1600)
    Which it does all the processing on.
    """
    def __init__(
     self, hsg_thing, parameter_dict=None, spectrometer_offset=None):
        # let the supers do the hard work of importing the
        # json dict and all that jazz
        super(HighSidebandCCDRaw, self).__init__(
            hsg_thing, parameter_dict=None, spectrometer_offset=None)
        self.ccd_data = np.genfromtxt(hsg_thing, delimiter=',').T
        self.proc_data = np.column_stack((self.gen_wavelengths(
            self.parameters["center_lambda"],
            self.parameters["grating"]), np.array(self.ccd_data[:, 1],
                                                  dtype=float)
            - np.median(self.ccd_data[:, 1]),
            np.ones_like(self.ccd_data[:, 1], dtype=float)))

        self.proc_data[:, 0] = 1239.84 / self.proc_data[:, 0]
        self.proc_data = np.flipud(self.proc_data)

    @staticmethod
    def gen_wavelengths(center_lambda, grating):
        '''
        This returns a 1600 element list of wavelengths for each pixel in the
            EMCCD based on grating and center wavelength

        grating = which grating, 1 or 2
        center = center wavelength in nanometers
        '''
        b = 0.75  # length of spectrometer, in m
        k = -1.0  # order looking at
        r = 16.0e-6  # distance between pixles on CCD

        if grating == 1:
            d = 1. / 1800000.
            gamma = 0.213258508834
            delta = 1.46389935365
        elif grating == 2:
            d = 1. / 1200000.
            gamma = 0.207412628027
            delta = 1.44998344749
        elif grating == 3:
            d = 1. / 600000.
            gamma = 0.213428934011
            delta = 1.34584754696
        else:
            print("What a dick, that's not a valid grating")
            return None

        center = center_lambda * 10 ** -9
        wavelength_list = np.arange(-799.0, 801.0)

        gammaNorm = np.cos((1 / 2) * gamma)

        KCenD = k * center / d

        repl = 2 * gammaNorm ** 2 * (
            2 - KCenD ** 2 + 2 * np.cos(gamma)) ** (1 / 2)

        # TODO: ensure that the below output matches that of the elder gods
        output = d/k * (
            (-1) * np.cos(
                delta + gamma - 1 * np.arccos(
                    (-1 / 4) * (1 / gammaNorm) ** 2 * (repl + KCenD * np.sin(
                        gamma)))
                + np.arctan(b ** (-1)
                            * (r * wavelength_list + b * np.cos(delta + gamma))
                            * (1 / np.sin(delta + gamma))))
            + (1 + (-1 / 16) * (1 / gammaNorm) ** 4 * (
                repl + d ** (-1) * k * center * np.sin(gamma)) ** 2)
            ** (1 / 2))
        output = (output + center) * 10 ** 9
        return output


"""
        # TODO: make this output from the elder gods self documenting
        is there a reason this must use d * k ** (-1) vs d/k
        output = d * k ** (-1) * ((-1) * np.cos(
            delta + gamma + (-1) * np.arccos(
                (-1 / 4) * (1 / np.cos((1 / 2) * gamma)) ** 2 *
                (2 * (np.cos((1 / 2) * gamma) ** 4 * (2 + (-1) * d ** (-2) * k
                      ** 2 * center ** 2 + 2 * np.cos(gamma)))
                 ** (1 / 2) + d ** (-1) * k * center * np.sin(gamma)))
            + np.arctan(b ** (-1) * (r * wavelength_list + b *
                        np.cos(delta + gamma)) * (1 / np.sin(delta + gamma))))
                        + (1 + (-1 / 16) * (1 / np.cos((1 / 2) * gamma)) ** 4 *
                           (2 * (np.cos((1 / 2) * gamma) ** 4 * (2 + (-1) * d
                            ** (-2) * k ** 2 * center ** 2 + 2 *
                            np.cos(gamma)))
                            ** (1 / 2) + d ** (-1) * k * center * np.sin(gamma)
                            ) ** 2) ** (1 / 2))
"""
