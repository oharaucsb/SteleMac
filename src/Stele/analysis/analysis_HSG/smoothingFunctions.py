import numpy as np
from scipy.optimize import curve_fit
import scipy.fftpack as fft
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=500)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688

    source:
    http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order + 1))
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def fft_filter(data, cutoffFrequency=1520, inspectPlots=False, tryFitting=False, freqSigma=50, ftol=1e-4,
               isInteractive=False):
    """
    Performs an FFT, then fits a peak in frequency around the
    input with the input width.
    If only data is given, it will cut off all frequencies above the default value.
    inspectPlots = True will plot the FFT and the filtering at each step, as well as the results
    tryFitting = True will try to fit the peak in frequency space centered at the cutoffFrequency
    and with a width of freqSigma, using the background function above. Will replace
    the peak with the background function. Feature not very well tested
    isInteractive: Will pop up interactive windows to move the cutoff frequency and view the
    FFT in real time. Requires pyqtgraph and PyQt4 installed (pyqt4 is standard with
    anaconda/winpython, but pyqtgraph is not)
    """
    # Make a copy so we can return the same thing
    retData = np.array(data)
    x = np.array(retData[:, 0])
    y = np.array(retData[:, -1])
    # Let's you place with zero padding.
    zeroPadding = len(x)
    N = len(x)

    if isInteractive:
        try:
            import pyqtgraph as pg
            from PyQt5 import QtCore, QtWidgets
        except:
            raise ImportError("Cannot do interactive plotting without pyqtgraph installed")

        # Need to make some basic classes fir signals and slots to make things simple
        class FFTWin(pg.PlotWindow):
            sigCutoffChanged = QtCore.pyqtSignal(object)
            sigClosed = QtCore.pyqtSignal()

            def __init__(self, x, y):
                super(FFTWin, self).__init__()
                # Plot the log of the data,
                # it breaks text boxes to do semilogy
                self.plotItem.plot(x, np.log10(y), pen='k')
                # The line for picking the cutoff
                # Connect signals so the textbox updates and the
                # realspace window can recalcualte the FFT
                self.line = pg.InfiniteLine(cutoffFrequency, movable=True)
                self.line.sigPositionChanged.connect(lambda x: self.sigCutoffChanged.emit(x.value()))
                self.line.sigPositionChanged.connect(self.updateText)
                self.addItem(self.line)
                # Set up the textbox so user knows the frequency
                # If this ends up being useful, may need
                # a way to set the cutoff manually
                self.text = pg.TextItem("{:.4f}".format(cutoffFrequency))
                self.addItem(self.text)
                self.text.setPos(min(x), max(np.log10(y)))

                # Cheap magic to get the close event
                # of the main window. Need to keep a reference
                # to the old function so that we can call it
                # to properly clean up afterwards
                self.oldCloseEvent = self.win.closeEvent
                self.win.closeEvent = self.closeEvent

            def updateText(self, val):
                self.text.setText("{:.4f}".format(val.value()))

            def closeEvent(self, ev):
                # Just emit that we've been closed and
                # pass it along to the window closer
                self.sigClosed.emit()
                self.oldCloseEvent(ev)

        class RealWin(pg.PlotWindow):
            sigClosed = QtCore.pyqtSignal()

            def __init__(self, data, fftWin):
                super(RealWin, self).__init__()
                # To connect signals from it
                self.fftWin = fftWin
                self.data = data

                # Start off with the FFT given by the original
                # inputted cutoff
                self.updatePlot(cutoffFrequency)

                # See above comments
                self.oldClose = self.win.closeEvent
                self.win.closeEvent = self.closeEvent
                fftWin.sigCutoffChanged.connect(self.updatePlot)
                # Close self if other window is closed
                fftWin.sigClosed.connect(self.win.close)

            def updatePlot(self, val):
                self.plotItem.clear()
                self.plotItem.plot(*self.data.T, pen=pg.mkPen('k', width=3))
                # Recursion! Call this same function to do the FFT
                newData = fft_filter(self.data, cutoffFrequency=val)
                self.plotItem.plot(*newData.T, pen=pg.mkPen('r', width=3))

            def closeEvent(self, ev):
                self.sigClosed.emit()
                try:
                    self.fftWin.win.close()
                except:
                    pass
                self.oldClose(ev)

        k = fft.fftfreq(zeroPadding, x[1] - x[0])
        Y = fft.fft(y, n=zeroPadding)
        # Make the windows
        fftWin = FFTWin(k, np.abs(Y))
        realWin = RealWin(np.array(retData), fftWin)
        realWin.show()
        # Need to pause the program until the frequency is selected
        # Done with this qeventloop.
        loop = QtCore.QEventLoop()
        realWin.sigClosed.connect(loop.exit)
        loop.exec_()
        # Return with the desired output value
        return fft_filter(retData, fftWin.line.value())

    if inspectPlots:
        plt.figure("Real Space")
        plt.plot(x, y, label="Input Data")

    # Replicate origin directy
    # http://www.originlab.com/doc/Origin-Help/Smooth-Algorithm
    # "rotate" the data set so it ends at 0,
    # enforcing a periodicity in the data. Otherwise
    # oscillatory artifacts result at the ends
    onePerc = int(0.01 * N)
    x1 = np.mean(x[:onePerc])
    x2 = np.mean(x[-onePerc:])
    y1 = np.mean(y[:onePerc])
    y2 = np.mean(y[-onePerc:])

    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1

    flattenLine = m * x + b
    y -= flattenLine

    if inspectPlots:
        plt.plot(x, y, label="Rotated Data")

    # Perform the FFT and find the appropriate frequency spacing
    k = fft.fftfreq(zeroPadding, x[1] - x[0])
    Y = fft.fft(y, n=zeroPadding)
    if inspectPlots:
        plt.figure("Frequency Space")
        plt.semilogy(k, np.abs(Y), label="Raw FFT")

    if tryFitting:
        try:
            # take +/- 4 sigma points around peak to fit to
            sl = np.abs(k - cutoffFrequency).argmin() + np.array([-1, 1]) * 10 * freqSigma / np.abs(k[0] - k[1])
            sl = slice(*[int(j) for j in sl])
            p0 = [cutoffFrequency,
                  np.abs(Y)[sl].max() * freqSigma,  # estimate the height baased on the max in the set
                  freqSigma,
                  0.14, 2e3, 1.1]  # magic test numbers, they fit the background well

            if inspectPlots:
                plt.semilogy(k[sl], gaussWithBackground(k[sl], *p0), label="Peak with initial values")
            p, _ = curve_fit(gaussWithBackground, k[sl], np.abs(Y)[sl], p0=p0, ftol=ftol)
            if inspectPlots:
                plt.semilogy(k[sl], gaussWithBackground(k[sl], *p), label="Fitted Peak")

            # Want to remove data within 5 sigma ( arb value... )
            st = int(p[0] - 5 * p[2])
            en = int(p[0] + 5 * p[2])

            # Find get the indices to remove.
            refitRangeIdx = np.argwhere((k > st) & (k < en))
            refitRangeIdxNeg = np.argwhere((k < -st) & (k > -en))

            # Replace the data with the backgroudn
            # Note: abuses the symmetry of the FFT of a real function
            # to get the negative side of the data
            Y[refitRangeIdx] = background(k[refitRangeIdx], *p[-2:])
            Y[refitRangeIdxNeg] = background(k[refitRangeIdx], *p[-2:])[::-1]
        except:
            print("ERROR: Trouble fitting the peak in frequency space.\n\t Defaulting to cutting off")

            # Assume cutoffFrequency was the peak, not the actual cutoff
            # Leaving it alone means half the peak would remain and the data
            # wouldn't really be smoothed
            cutoffFrequency -= 5 * freqSigma

            # Reset this so the next part gets called
            tryFitting = False

    # "if not" instead of "else" because if the above
    # fitting fails, we can default to the sharp cutoff
    if not tryFitting:
        # Define where to remove the data
        st = cutoffFrequency
        en = int(max(k)) + 1

        # Find the indices to remove the data
        refitRangeIdx = np.argwhere((k > st) & (k < en))
        refitRangeIdxNeg = np.argwhere((k < -st) & (k > -en))

        # Kill it all after the cutoff
        Y[refitRangeIdx] = 0
        Y[refitRangeIdxNeg] = 0

        smoothIdx = np.argwhere((-st < k) & (k < st))
        smoothr = -1. / cutoffFrequency ** 2 * k[smoothIdx] ** 2 + 1

        Y[smoothIdx] *= smoothr

    if inspectPlots:
        plt.plot(k, np.abs(Y), label="FFT with removed parts")
        a = plt.legend()
        a.draggable(True)

    # invert the FFT
    y = fft.ifft(Y, n=zeroPadding)

    # unshift the data
    y += flattenLine

    # using fft, not rfft, so data may have some
    # complex parts. But we can assume they'll be negligible and
    # remove them
    # ( Safer to use np.real, not np.abs? )
    # Need the [:len] to remove zero-padded stuff
    y = np.abs(y)[:len(x)]

    if inspectPlots:
        plt.figure("Real Space")
        print(x.size, y.size)
        plt.plot(x, y, label="Smoothed Data")
        a = plt.legend()
        a.draggable(True)

    retData[:, 0] = x
    retData[:, -1] = y
    return retData


def low_pass_filter(x_vals, y_vals, cutoff, inspectPlots=True):
    """
    Replicate origin directy
    http://www.originlab.com/doc/Origin-Help/Smooth-Algorithm
    "rotate" the data set so it ends at 0,
    enforcing a periodicity in the data. Otherwise
    oscillatory artifacts result at the ends

    This uses a 50th order Butterworth filter.
    """
    x_vals, y_vals = fourier_prep(x_vals, y_vals)
    if inspectPlots:
        plt.figure("Real Space")
        plt.plot(x_vals, y_vals, label="Non-nan Data")

    zeroPadding = len(x_vals)
    # print "zero padding", zeroPadding  # This needs to be this way because truncation is bad and actually zero padding
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
    band_start = cutoff
    band_end = int(max(abs(x_fourier))) + 1

    '''
    # Find the indices to remove the data
    refitRangeIdx = np.argwhere((x_fourier > band_start) & (x_fourier <= band_end))
    refitRangeIdxNeg = np.argwhere((x_fourier < -band_start) & (x_fourier >= -band_end))

    #print "x_fourier", x_fourier[795:804]
    #print "max(x_fourier)", max(x_fourier)
    #print "refitRangeIdxNeg", refitRangeIdxNeg[:-400]

    # Kill it all after the cutoff
    y_fourier[refitRangeIdx] = 0
    y_fourier[refitRangeIdxNeg] = 0

    # This section does a square filter on the remaining code.
    smoothIdx = np.argwhere((-band_start < x_fourier) & (x_fourier < band_start))
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


def high_pass_filter(x_vals, y_vals, cutoff, inspectPlots=True):
    """
    Replicate origin directy
    http://www.originlab.com/doc/Origin-Help/Smooth-Algorithm
    "rotate" the data set so it ends at 0,
    enforcing a periodicity in the data. Otherwise
    oscillatory artifacts result at the ends

    This uses a 50th order Butterworth filter.
    """
    x_vals, y_vals = fourier_prep(x_vals, y_vals)
    if inspectPlots:
        plt.figure("Real Space")
        plt.plot(x_vals, y_vals, label="Non-nan Data")

    zeroPadding = len(x_vals)
    print("zero padding", zeroPadding)  # This needs to be this way because truncation is bad and actually zero padding
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
    band_start = cutoff
    band_end = int(max(abs(x_fourier))) + 1

    '''
    # Find the indices to remove the data
    refitRangeIdx = np.argwhere((x_fourier > band_start) & (x_fourier <= band_end))
    refitRangeIdxNeg = np.argwhere((x_fourier < -band_start) & (x_fourier >= -band_end))

    #print "x_fourier", x_fourier[795:804]
    #print "max(x_fourier)", max(x_fourier)
    #print "refitRangeIdxNeg", refitRangeIdxNeg[:-400]

    # Kill it all after the cutoff
    y_fourier[refitRangeIdx] = 0
    y_fourier[refitRangeIdxNeg] = 0

    # This section does a square filter on the remaining code.
    smoothIdx = np.argwhere((-band_start < x_fourier) & (x_fourier < band_start))
    smoothr = -1 / band_start**2 * x_fourier[smoothIdx]**2 + 1

    y_fourier[smoothIdx] *= smoothr
    '''

    print(abs(y_fourier[-10:]))
    butterworth = 1 - np.sqrt(1 / (1 + (x_fourier / cutoff) ** 50))
    y_fourier *= butterworth

    if inspectPlots:
        plt.plot(x_fourier, np.abs(y_fourier), label="FFT with removed parts")
        a = plt.legend()
        a.draggable(True)
        print("y_fourier", len(y_fourier))

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
        print(x_vals.size, y_vals.size)
        plt.plot(x_vals, y_vals, label="Smoothed Data")
        a = plt.legend()
        a.draggable(True)

    return np.column_stack((x_vals, y_vals))


def band_pass_filter(x_vals, y_vals, cutoff, inspectPlots=True):
    """
    Replicate origin directy
    http://www.originlab.com/doc/Origin-Help/Smooth-Algorithm
    "rotate" the data set so it ends at 0,
    enforcing a periodicity in the data. Otherwise
    oscillatory artifacts result at the ends

    This uses a 50th order Butterworth filter.
    """
    x_vals, y_vals = fourier_prep(x_vals, y_vals)
    if inspectPlots:
        plt.figure("Real Space")
        plt.plot(x_vals, y_vals, label="Non-nan Data")

    zeroPadding = len(x_vals)
    print("zero padding", zeroPadding)  # This needs to be this way because truncation is bad and actually zero padding
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
    band_start = cutoff
    band_end = int(max(abs(x_fourier))) + 1

    '''
    # Find the indices to remove the data
    refitRangeIdx = np.argwhere((x_fourier > band_start) & (x_fourier <= band_end))
    refitRangeIdxNeg = np.argwhere((x_fourier < -band_start) & (x_fourier >= -band_end))

    #print "x_fourier", x_fourier[795:804]
    #print "max(x_fourier)", max(x_fourier)
    #print "refitRangeIdxNeg", refitRangeIdxNeg[:-400]

    # Kill it all after the cutoff
    y_fourier[refitRangeIdx] = 0
    y_fourier[refitRangeIdxNeg] = 0

    # This section does a square filter on the remaining code.
    smoothIdx = np.argwhere((-band_start < x_fourier) & (x_fourier < band_start))
    smoothr = -1 / band_start**2 * x_fourier[smoothIdx]**2 + 1

    y_fourier[smoothIdx] *= smoothr
    '''

    print(abs(y_fourier[-10:]))
    butterworth = 1 - np.sqrt(1 / (1 + (x_fourier / cutoff[0]) ** 50))
    butterworth *= np.sqrt(1 / (1 + (x_fourier / cutoff[1]) ** 50))
    y_fourier *= butterworth

    if inspectPlots:
        plt.plot(x_fourier, np.abs(y_fourier), label="FFT with removed parts")
        a = plt.legend()
        a.draggable(True)
        print("y_fourier", len(y_fourier))

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
        print(x_vals.size, y_vals.size)
        plt.plot(x_vals, y_vals, label="Smoothed Data")
        a = plt.legend()
        a.draggable(True)

    return np.column_stack((x_vals, y_vals))
