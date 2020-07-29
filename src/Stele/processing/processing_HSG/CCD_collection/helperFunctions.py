import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from .processing_HSG import helperFunctions


def calc_laser_frequencies(spec, nir_units="eV", thz_units="eV",
                           bad_points=-2, inspect_plots=False):
    """
    Calculate the NIR and FEL frequency for a spectrum
    :param spec: HSGCCD object to fit
    :type spec: HighSidebandCCD
    :param nir_units: str of desired units.
        Options: wavenumber, eV, meV, THz, GHz, nm
    :param thz_units: str of desired units.
        Options: wavenumber, eV, meV, THz, GHz, nm
    :param bad_points: How many bad points which shouldn't be used
        to calculate the frequencies (generally because the last
        few points are noisy and unreliable)
    :return: <NIR freq>, <THz freq>
    """
    if not hasattr(spec, "sb_results"):
        spec.guess_sidebands()
        spec.fit_sidebands()

    sidebands = spec.sb_results[:, 0]
    locations = spec.sb_results[:, 1]
    errors = spec.sb_results[:, 2]
    try:
        p = np.polyfit(sidebands[1:bad_points],
                       # This is 1 because the peak picker function was
                       #    calling the 10th order the 9th
                       locations[1:bad_points], deg=1)
    except TypeError:
        # if there aren't enough sidebands to fit, give -1
        p = [-1, -1]

    NIRfreq = p[1]
    THzfreq = p[0]

    if inspect_plots:
        plt.figure("Frequency Fit")
        plt.errorbar(sidebands, locations, errors, marker='o')
        plt.errorbar(sidebands[:bad_points], locations[:bad_points],
                     errors[:bad_points], marker='o')
        plt.plot(sidebands, np.polyval(p, sidebands))

    converter = {
        "eV": lambda x: x,
        "meV": lambda x: 1000. * x,
        "wavenumber": lambda x: 8065.6 * x,
        "THz": lambda x: 241.80060 * x,
        "GHz": lambda x: 241.80060 * 1e3 * x,
        "nm": lambda x: 1239.83 / x
    }

    freqNIR = converter.get(nir_units, converter["eV"])(NIRfreq)
    freqTHz = converter.get(thz_units, converter["eV"])(THzfreq)

    return freqNIR, freqTHz


def low_pass_filter(x_vals, y_vals, cutoff, inspectPlots=True):
    """
    Replicate origin directy
    http://www.originlab.com/doc/Origin-Help/Smooth-Algorithm
    "rotate" the data set so it ends at 0,
    enforcing a periodicity in the data. Otherwise
    oscillatory artifacts result at the ends

    This uses a 50th order Butterworth filter.
    """
    x_vals, y_vals = helperFunctions.fourier_prep(x_vals, y_vals)
    if inspectPlots:
        plt.figure("Real Space")
        plt.plot(x_vals, y_vals, label="Non-nan Data")

    zeroPadding = len(x_vals)
    # print "zero padding", zeroPadding
    # Required because truncation is bad & actually zero padding
    N = len(x_vals)
    onePerc = int(0.01 * N)
    x1 = np.mean(x_vals[:onePerc])
    x2 = np.mean(x_vals[-onePerc:])
    y1 = np.mean(y_vals[:onePerc])
    y2 = np.mean(y_vals[-onePerc:])

    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1

    flattenLine = m * x_vals + b
    y_vals -= flattenLine

    if inspectPlots:
        plt.figure("Real Space")
        plt.plot(x_vals, y_vals, label="Rotated Data")

    # even_data = np.column_stack((x_vals, y_vals))
    # Perform the FFT and find the appropriate frequency spacing
    x_fourier = fft.fftfreq(zeroPadding, x_vals[1] - x_vals[0])
    y_fourier = fft.fft(y_vals)  # , n=zeroPadding)

    if inspectPlots:
        plt.figure("Frequency Space")
        plt.semilogy(x_fourier, np.abs(y_fourier), label="Raw FFT")

    # Define where to remove the data
    # band_start = cutoff
    # band_end = int(max(abs(x_fourier))) + 1

    '''
    # Find the indices to remove the data
    refitRangeIdx = np.argwhere((x_fourier > band_start)
        & (x_fourier <= band_end))
    refitRangeIdxNeg = np.argwhere((x_fourier < -band_start)
        & (x_fourier >= -band_end))

    #print "x_fourier", x_fourier[795:804]
    #print "max(x_fourier)", max(x_fourier)
    #print "refitRangeIdxNeg", refitRangeIdxNeg[:-400]

    # Kill it all after the cutoff
    y_fourier[refitRangeIdx] = 0
    y_fourier[refitRangeIdxNeg] = 0

    # This section does a square filter on the remaining code.
    smoothIdx = np.argwhere((-band_start < x_fourier)
        & (x_fourier < band_start))
    smoothr = -1 / band_start**2 * x_fourier[smoothIdx]**2 + 1

    y_fourier[smoothIdx] *= smoothr
    '''

    # print abs(y_fourier[-10:])
    butterworth = np.sqrt(1 / (1 + (x_fourier / cutoff) ** 100))
    y_fourier *= butterworth

    if inspectPlots:
        plt.plot(x_fourier, np.abs(y_fourier), label="FFT with removed parts")
        a = plt.legend()
        a.draggable(True)
        # print "y_fourier", len(y_fourier)

    # invert the FFT
    y_vals = fft.ifft(y_fourier, n=zeroPadding)

    # using fft, not rfft, so data may have some
    # complex parts. But we can assume they'll be negligible and
    # remove them
    # ( Safer to use np.real, not np.abs? )
    # Need the [:len] to remove zero-padded stuff
    y_vals = y_vals[:len(x_vals)]
    # unshift the data
    y_vals += flattenLine
    y_vals = np.abs(y_vals)

    if inspectPlots:
        plt.figure("Real Space")
        # print x_vals.size, y_vals.size
        plt.plot(x_vals, y_vals, linewidth=3, label="Smoothed Data")
        a = plt.legend()
        a.draggable(True)

    return np.column_stack((x_vals, y_vals))


# photonConverter[A][B](x):
#    convert x from A to B.
photon_converter = {
    "nm": {
        "nm": lambda x: x,
        "eV": lambda x: 1239.84/x,
        "wavenumber": lambda x: 10000000./x},
    "eV": {
        "nm": lambda x: 1239.84/x,
        "eV": lambda x: x,
        "wavenumber": lambda x: 8065.56 * x},
    "wavenumber": {
        "nm": lambda x: 10000000./x,
        "eV": lambda x: x/8065.56,
        "wavenumber": lambda x: x}
}
