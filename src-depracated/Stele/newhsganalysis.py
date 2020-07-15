# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:16:25 2015

@author: hbanks

"""


import os
import io
import glob
import errno
import copy
import json
import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import itertools as itt
np.set_printoptions(linewidth=500)


####################
# Fitting functions
####################
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

####################
# Collection functions
####################
def hsg_combine_spectra(spectra_list, verbose = False, **kwargs):
    """
    This function is all about smooshing different parts of the same hsg
    spectrum together.  It takes a list of HighSidebandCCD spectra and turns the
    zeroth spec_step into a FullHighSideband object.  It then uses the function
    stitch_hsg_dicts over and over again for the smooshing.

    Input:
    spectra_list = list of HighSidebandCCD objects that have sideband spectra
                   larger than the spectrometer can see.

    Returns:
    good_list = A list of FullHighSideband objects that have been combined as
                much as can be.

    :param spectra_list: randomly-ordered list of HSG spectra, some of which can be stitched together
    :type spectra_list: List of HighSidebandCCD objects
    kwargs gets passed onto add_item
    :return: fully combined list of full hsg spectra.  No PMT business yet.
    :rtype: list of FullHighSideband
    """
    good_list = []
    spectra_list = spectra_list.copy()
    spectra_list.sort(key=lambda x: x.parameters["spec_step"])

    # keep a dict for each series' spec step
    # This allows you to combine spectra whose spec steps
    # change by values other than 1 (2, if you skip, or 0.5 if you
    # decide to insert things, or arbitary strings)
    spec_steps = {}

    for elem in spectra_list:
        # if verbose:
        #     print "Spec_step is", elem.parameters["spec_step"]
        current_steps = spec_steps.get(elem.parameters["series"], [])
        current_steps.append(elem.parameters["spec_step"])
        spec_steps[elem.parameters["series"]] = current_steps
    if verbose:
        print("I found these spec steps for each series:")
        print("\n\t".join("{}: {}".format(*ii) for ii in spec_steps.items()))

    # sort the list of spec steps
    for series in spec_steps:
        spec_steps[series].sort()

    same_freq = lambda x,y: x.parameters["fel_lambda"] == y.parameters["fel_lambda"]

    for index in range(len(spectra_list)):
        try:
            temp = spectra_list.pop(0)
            if verbose:
                print("\nStarting with this guy", temp, "\n")
        except:
            break

        good_list.append(FullHighSideband(temp))

        counter = 1
        temp_list = list(spectra_list)
        for piece in temp_list:
            if verbose:
                print("\tchecking this spec_step", piece.parameters["spec_step"], end=' ')
                print(", the counter is", counter)
            if not same_freq(piece, temp):
                if verbose:
                    print("\t\tnot the same fel frequencies ({} vs {})".format(piece.parameters["fel_lambda"], temp.parameters["fel_lambda"]))
                continue
            if temp.parameters["series"] == piece.parameters["series"]:
                if piece.parameters["spec_step"] == spec_steps[temp.parameters["series"]][counter]:
                    if verbose:
                        print("I found this one", piece)
                    counter += 1
                    good_list[-1].add_CCD(piece, verbose=verbose, **kwargs)
                    spectra_list.remove(piece)
                else:
                    print("\t\tNot the right spec step?", type(piece.parameters["spec_step"]))

            else:
                if verbose:
                    print("\t\tNot the same series ({} vs {}".format(
                        piece.parameters["series"],temp.parameters["series"]))
        good_list[-1].make_results_array()
    return good_list

def hsg_combine_spectra_arb_param(spectra_list, param_name="series", verbose = False):
    """
    This function is all about smooshing different parts of the same hsg
    spectrum together.  It takes a list of HighSidebandCCD spectra and turns the
    zeroth spec_step into a FullHighSideband object.  It then uses the function
    stitch_hsg_dicts over and over again for the smooshing.

    This is different than hsg_combine_spectra in that you pass which
    criteria distinguishes the files to be the "same". Since it can be any arbitrary
    value, things won't be exactly the same (field strength will never be identical
    between images). It will start with the first (lowest) spec step, then compare the
    number of images in the next step. Whichever has

    Input:
    spectra_list = list of HighSidebandCCD objects that have sideband spectra
                   larger than the spectrometer can see.

    Returns:
    good_list = A list of FullHighSideband objects that have been combined as
                much as can be.

    :param spectra_list: randomly-ordered list of HSG spectra, some of which can be stitched together
    :type spectra_list: list of HighSidebandCCD
    :return: fully combined list of full hsg spectra.  No PMT business yet.
    :rtype: list of FullHighSideband
    """
    if not spectra_list:
        raise RuntimeError("Passed an empty spectra list!")
    if isinstance(param_name, list):
        # if you pass two things because the param you want
        # is in a dict (e.g. field strength has mean/std)
        # do it that way
        param_name_list = list(param_name)
        paramGetter = lambda x: x.parameters[param_name_list[0]][param_name_list[1]]
        param_name = param_name[0]
    elif isinstance(spectra_list[0].parameters[param_name], dict):
        paramGetter = lambda x: x.parameters[param_name]["mean"]
    else:
        paramGetter = lambda x: x.parameters[param_name]

    good_list = []
    spectra_list.sort(key=lambda x: x.parameters["spec_step"])

    # keep a dict for each spec step.
    spec_steps = {}
    for elem in spectra_list:
        if verbose:
            print("Spec_step is", elem.parameters["spec_step"])
        current_steps = spec_steps.get(elem.parameters["spec_step"], [])
        current_steps.append(elem)
        spec_steps[elem.parameters["spec_step"]] = current_steps


    # Next, loop over all of the elements. For each element, if it has not
    # already been added to a spectra, look at all of the combinations from
    # other spec steps to figure out which has the smallest overall deviation
    # to make a new full spectrum
    good_list = []
    already_added = set()
    for elem in spectra_list:
        if elem in already_added: continue
        already_added.add(elem)
        good_list.append(FullHighSideband(elem))

        other_spec_steps = [v for k, v in list(spec_steps.items()) if
                            k != good_list[-1].parameters["spec_step"]]
        min_distance = np.inf
        cur_value = paramGetter(good_list[-1])
        best_match = None
        for comb in itt.product(*other_spec_steps):
            new_values = list(map(paramGetter, comb))
            all_values = new_values + [cur_value]

            if np.std(all_values) < min_distance:
                min_distance = np.std(all_values)
                best_match = list(comb)

        if best_match is None:
            raise RuntimeError("No matches found. Empty lists passed?")

        best_values = list(map(paramGetter, best_match))
        for spec in best_match:
            print("Adding new spec step\n\tStarted with spec={},series={}".format(
                good_list[-1].parameters["spec_step"],good_list[-1].parameters["series"]
            ))
            print("\tAdding with spec={},series={}\n".format(
                spec.parameters["spec_step"],
                spec.parameters["series"]
            ))
            print("\n\nfirst SBs:\n", good_list[-1].sb_results)
            print("\n\nsecond SBs:\n", spec.sb_results)
            good_list[-1].add_CCD(spec, True)
            print("\n\nEnding SBs:\n", good_list[-1].sb_results)

            already_added.add(spec)
        best_match.append(good_list[-1])
        best_values.append(cur_value)
        new_value = np.mean(best_values)
        new_std = np.std(best_values)

        if isinstance(good_list[-1].parameters[param_name], dict):
            best_values = np.array([x.parameters[param_name]["mean"] for x in best_match])
            best_std = np.array([x.parameters[param_name]["std"] for x in best_match])
            new_value = np.average(best_values, weights = best_std)
            new_std = np.sqrt(np.average((best_values-new_value)**2, weights=best_std))

        good_list[-1].parameters[param_name] = {
            "mean": new_value,
            "std": new_std
        }
    return good_list

def pmt_sorter(folder_path, plot_individual = True):
    """
    This function will be fed a folder with a bunch of PMT data files in it.
    The folder should contain a bunch of spectra with at least one sideband in
    them, each differing by the series entry in the parameters dictionary.

    This function will return a list of HighSidebandPMT objects.

    :param folder_path: Path to a folder containing a bunch of PMT data, can be
                        part of a parameter sweep
    :type folder_path: str
    :param plot_individual: Whether to plot each sideband itself
    :return: A list of all the possible hsg pmt spectra, organized by series tag
    :rtype: list of HighSidebandPMT
    """
    file_list = glob.glob(os.path.join(folder_path, '*[0-9].txt'))

    pmt_list = []

    plot_sb = lambda x: None

    if plot_individual:
        plt.figure("PMT data")
        def plot_sb(spec):
            spec = copy.deepcopy(spec)
            spec.process_sidebands()
            elem = spec.sb_dict[spec.initial_sb]
            plt.errorbar(elem[:, 0], elem[:, 1], elem[:, 2],
                     marker='o',
                     label="{} {}, {}.{} ".format(
                         spec.parameters["series"], spec.initial_sb,
                         spec.parameters["pm_hv"],
                         't' if spec.parameters.get("photon counted", False) else 'f')
                         )

    for sb_file in file_list:
        temp = HighSidebandPMT(sb_file)
        plot_sb(temp)
        try:
            for pmt_spectrum in pmt_list:  # pmt_spectrum is a pmt object
                if temp.parameters['series'] == pmt_spectrum.parameters['series']:
                    pmt_spectrum.add_sideband(temp)
                    break
            else:  # this will execute IF the break was NOT called
                pmt_list.append(temp)
        except:
            pmt_list.append(temp)
    # for sb_file in file_list:
    #     with open(sb_file,'rU') as f:
    #         param_str = ''
    #         line = f.readline()
    #         line = f.readline()
    #         while line[0] == '#':
    #             param_str += line[1:]
    #             line = f.readline()
    #
    #         parameters = json.loads(param_str)
    #     try:
    #         for pmt_spectrum in pmt_list: # pmt_spectrum is a pmt object?
    #             if parameters['series'] == pmt_spectrum.parameters['series']:
    #                 pmt_spectrum.add_sideband(sb_file)
    #                 break
    #         else: # this will execute IF the break was NOT called
    #             pmt_list.append(HighSidebandPMT(sb_file))
    #     except:
    #         pmt_list.append(HighSidebandPMT(sb_file))

    for pmt_spectrum in pmt_list:
        pmt_spectrum.process_sidebands()
    return pmt_list

def stitch_abs_results(main, new):
    raise NotImplementedError

def hsg_combine_qwp_sweep(path, loadNorm = True, save = False, verbose=False,
                          skipOdds = True):
    """
    Given a path to data taken from rotating the QWP (doing polarimetry),
    process the data (fit peaks), and parse it into a matrix of sb strength vs
    QWP angle vs sb number.

    By default, saves the file into "Processed QWP Dependence"

    Return should be passed directly into fitting

         -1     |     SB1     |   SB1  |     SB2     |   SB2  |    ...    |   ...  |     SBn     |   SBn  |
      angle1    | SB Strength | SB err | SB Strength | SB Err |
      angle2    |     ...     |    .   |
      .
      .
      .

    :param path: Path to load
    :param loadNorm: if true, load the normalized data
    :param save: Save the processed file or not
    :param verbose:
    :param skipOdds: Passed on to save sweep; determine whether or not to save
            odd orders. Generally, odds are artifacts and I don't want
            them messing up the data, so default to True.
    :return:
    """
    def getData(fname):
        """
        Helper function for loading the data and getting the header information for incident NIR stuff
        :param fname:
        :return:
        """
        if isinstance(fname, str):
            if loadNorm:
                ending = "_norm.txt"
            else:
                ending = "_snip.txt"
            header = ''
            with open(os.path.join("Processed QWP Dependence", fname + ending)) as fh:
                ln = fh.readline()
                while ln[0] == '#':
                    header += ln[1:]
                    ln = fh.readline()
            data = np.genfromtxt(os.path.join("Processed QWP Dependence", fname + ending),
                                 delimiter=',', dtype=str)
        if isinstance(fname, io.BytesIO):
            header = b''
            ln = fname.readline()
            while ln.decode()[0] == '#':
                header += ln[1:]
                ln = fname.readline()
            fname.seek(0)
            data = np.genfromtxt(fname,
                                 delimiter=',', dtype=str)

        header = json.loads(header)
        return data, float(header["lAlpha"]), float(header["lGamma"]), float(header["nir"]), float(header["thz"])
        ######### End getData

    try:
        sbData, lAlpha, lGamma, nir, thz = getData(path)
    except:
        # Do the processing on all the files
        specs = proc_n_plotCCD(path, keep_empties=True, verbose=verbose)

        for sp in specs:
            try:
                sp.parameters["series"] = round(float(sp.parameters["rotatorAngle"]), 2)
            except KeyError:
                # Old style of formatting
                sp.parameters["series"] = round(float(sp.parameters["detectorHWP"]), 2)
        specs = hsg_combine_spectra(specs, ignore_weaker_lowers=False)
        if not save:
            # If you don't want to save them, set everything up for doing Bytes objects
            # to replacing saving files
            full, snip, norm = io.BytesIO(), io.BytesIO(), io.BytesIO()
            if "nir_pola" not in specs[0].parameters:
                # in the olden days. Force them. Hopefully making them outside of ±360
                # makes it obvious
                specs[0].parameters["nir_pola"] = 361
                specs[0].parameters["nir_polg"] = 361
            keyName = "rotatorAngle"
            if keyName not in specs[0].parameters:
                # from back before I changed the name
                keyName = "detectorHWP"

            save_parameter_sweep(specs, [full, snip, norm], None,
                                     keyName, "deg", wanted_indices=[3, 4],
                                     header_dict={
                                         "lAlpha": specs[0].parameters["nir_pola"],
                                         "lGamma": specs[0].parameters["nir_polg"],
                                         "nir": specs[0].parameters["nir_lambda"],
                                         "thz": specs[0].parameters["fel_lambda"], },
                                 only_even=skipOdds)

            if loadNorm:
                sbData, lAlpha, lGamma, nir, thz = getData(norm)
            else:
                sbData, lAlpha, lGamma, nir, thz = getData(snip)
        else:
            save_parameter_sweep(specs, os.path.basename(path), "Processed QWP Dependence",
                                 "rotatorAngle", "deg", wanted_indices=[3, 4],
                                 header_dict={
                                     "lAlpha": specs[0].parameters["nir_pola"],
                                     "lGamma": specs[0].parameters["nir_polg"],
                                     "nir": specs[0].parameters["nir_lambda"],
                                     "thz": specs[0].parameters["fel_lambda"], },
                                 only_even=skipOdds)
            sbData, lAlpha, lGamma, nir, thz = getData(os.path.basename(path))

    laserParams = {
        "lAlpha": lAlpha,
        "lGamma": lGamma,
        "nir": nir,
        "thz": thz
    }

    # get which sidebands were found in this data set
    # first two rows are origin header, second is sideband number
    # (and empty strings, which is why the "if ii" below, to prevent
    # ValueErrors on int('').
    foundSidebands = np.array(sorted([float(ii) for ii in set(sbData[2]) if ii]))

    # Remove first 3 rows, which are strings for origin header, and cast it to floats
    sbData = sbData[3:].astype(float)

    # double the sb numbers (to account for sb strength/error) and add a dummy
    # number so the array is the same shape
    foundSidebands = np.insert(foundSidebands, range(len(foundSidebands)), foundSidebands)
    foundSidebands = np.insert(foundSidebands, 0, -1)
    return laserParams, np.row_stack((foundSidebands, sbData))

def makeCurve(eta, isVertical):
    """

    :param eta: QWP retardance at the wavelength
    :return:
    """
    cosd = lambda x: np.cos(x * np.pi / 180)
    sind = lambda x: np.sin(x * np.pi / 180)
    eta = eta * 2 * np.pi
    if isVertical:
        # vertical polarizer
        def analyzerCurve(x, *S):
            S0, S1, S2, S3 = S
            return S0-S1/2*(1+np.cos(eta)) \
                   + S3*np.sin(eta)*sind(2*x) \
                   + S1/2*(np.cos(eta)-1)*cosd(4*x) \
                   + S2/2*(np.cos(eta)-1)*sind(4*x)
    else:
        # vertical polarizer
        def analyzerCurve(x, *S):
            S0, S1, S2, S3 = S
            return S0+S1/2*(1+np.cos(eta)) \
                   - S3*np.sin(eta)*sind(2*x) \
                   + S1/2*(1-np.cos(eta))*cosd(4*x) \
                   + S2/2*(1-np.cos(eta))*sind(4*x)
    return analyzerCurve

def proc_n_fit_qwp_data(data, laserParams = dict(), wantedSBs = None, vertAnaDir = True, plot=False,
                        save = False, plotRaw = lambda sbidx, sbnum: False, series = '', eta=None,
                        **kwargs):
    """
    Fit a set of sideband data vs QWP angle to get the stoke's parameters
    :param data: data in the form of the return of hsg_combine_qwp_sweep
    :param laserParams: dictionary of the parameters of the laser, the angles and frequencies. See function for
                expected keys. I don't think the errors are used (except for plotting?), or the wavelengths (but
                left in for potential future use (wavelength dependent stuff?))
    :param wantedSBs: List of the wanted sidebands to fit out.
    :param vertAnaDir: direction of the analzyer. True if vertical, false if horizontal.
    :param plot: True/False to plot alpha/gamma/dop. Alternatively, a list of "a", "g", "d" to only plot selected ones
    :param save: filename to save the files. Accepts BytesIO
    :param plotRaw: callable that takes an index of the sb and sb number, returns true to plot the raw curve
    :param series: a string to be put in the header for the origin files
    :param eta: a function to call to calculate the desired retardance. Input will be the SB order.

    if saveStokes is in kwargs and False, it will not save the stokes parameters, since I rarely actually use them.
    :return:
    """
    defaultLaserParams = {
        "lAlpha": 90,
        "ldAlpha": 0.2,
        "lGamma": 0.0,
        "ldGamma": 0.2,
        "lDOP": 1,
        "ldDOP": 0.02,
        "nir": 765.7155,
        "thz": 21.1
    }
    defaultLaserParams.update(laserParams)
    lAlpha, ldAlpha, lGamma, ldGamma, lDOP, ldDOP = defaultLaserParams["lAlpha"], \
                                                    defaultLaserParams["ldAlpha"], \
                                                    defaultLaserParams["lGamma"], \
                                                    defaultLaserParams["ldGamma"], \
                                                    defaultLaserParams["lDOP"], \
                                                    defaultLaserParams["ldDOP"]
    allSbData = data
    angles = allSbData[1:, 0]

    # angles += -5
    # print("="*20)
    # print("\n"*3)
    # print("             WARNING")
    # print("\n"*3)
    # print("ANGLES HAVE BEEN MANUALLY OFFEST IN proc_n_fit_qwp_data")
    # print("\n"*3)
    # print("="*20)

    allSbData = allSbData[:, 1:] # trim out the angles

    if wantedSBs is None:
        # set to get rid of duplicates, 1: to get rid of the -1 used for
        # getting arrays the right shape
        wantedSBs = set(allSbData[0, 1:])

    if eta is None:
        """
        It might be easier for the end user to do this by passing eta(wavelength) instead of eta(sborder),
        but then this function would need to carry around wavelengths, which is extra work. It could convert
        between NIR/THz wavelengths to SB order, but it's currently unclear whether you'd rather use what the WS6
        claims, or what the sidebands say, and you'd probably want to take the extra step to ensure the SB fit rseults
        if using the spectromter wavelengths. In general, if you have a function as etal(wavelength), you'd probably
        want to pass this as
        eta = lambda x: etal(1239.84/(nirEv + x*THzEv))
        assuming nirEv/THzEv are the photon energies of the NIR/THz.
        """
        eta = lambda x: 0.25

    # allow pasing a flag it ignore odds. I think I generally do, so set it to
    # default to True
    skipOdds = kwargs.get("skip_odds", True)

    # Make an array to keep all of the sideband information.
    # Start it off by keeping the NIR information (makes for easier plotting into origin)
    sbFits = [[0] + [-1] * 8 + [lAlpha, ldAlpha, lGamma, ldGamma, lDOP, ldDOP]]
    # Also, for convenience, keep a dictionary of the information.
    # This is when I feel like someone should look at porting this over to pandas
    sbFitsDict = {}
    sbFitsDict["S0"] = [[0, -1, -1]]
    sbFitsDict["S1"] = [[0, -1, -1]]
    sbFitsDict["S2"] = [[0, -1, -1]]
    sbFitsDict["S3"] = [[0, -1, -1]]
    sbFitsDict["alpha"] = [[0, lAlpha, ldAlpha]]
    sbFitsDict["gamma"] = [[0, lGamma, ldGamma]]
    sbFitsDict["DOP"] = [[0, lDOP, ldDOP]]

    # Iterate over all sb data. Skip by 2 because error bars are included
    for sbIdx in range(0, allSbData.shape[1], 2):
        sbNum = allSbData[0, sbIdx]
        if sbNum not in wantedSBs: continue
        if skipOdds and sbNum%2: continue
        # if verbose:
        #     print("\tlooking at sideband", sbNum)
        sbData = allSbData[1:, sbIdx]
        sbDataErr = allSbData[1:, sbIdx + 1]

        # try:
        #     p0 = sbFits[-1][1:8:2]
        # except:
        #     p0 = [1, 1, 0, 0]
        p0 = [1, 1, 0, 0]

        etan = eta(sbNum)
        try:
            p, pcov = curve_fit(makeCurve(etan, vertAnaDir), angles, sbData, p0=p0)
        except ValueError:
            # This is getting tossed around, especially when looking at noisy data,
            # especially with the laser line, and it's fitting erroneous values.
            # Ideally, I should be cutting this out and not even returning them,
            # but that's immedaitely causing
            p = np.nan*np.array(p0)
            pcov = np.eye(len(p))


        if plot and plotRaw(sbIdx, sbNum):
            # pg.figure("{}: sb {}".format(dataName, sbNum))
            plt.figure("All Curves")
            plt.errorbar(angles, sbData, sbDataErr, 'o-', name=f"{series}, {sbNum}")
            # plt.plot(angles, sbData,'o-', label="Data")
            fineAngles = np.linspace(angles.min(), angles.max(), 300)
            # plt.plot(fineAngles,
            #         makeCurve(eta, "V" in dataName)(fineAngles, *p0), name="p0")
            plt.plot(fineAngles,
                    makeCurve(etan, vertAnaDir)(fineAngles, *p))
            # plt.show()
            plt.ylim(0, 1)
            plt.xlim(0, 360)
            plt.ylabel("Normalized Intensity")
            plt.xlabel("QWP Angle (&theta;)")
            print(f"\t{series} {sbNum}, p={p}")


        # get the errors
        d = np.sqrt(np.diag(pcov))
        thisData = [sbNum] + list(p) + list(d)
        d0, d1, d2, d3 = d
        S0, S1, S2, S3 = p
        # reorder so errors are after values
        thisData = [thisData[i] for i in [0, 1, 5, 2, 6, 3, 7, 4, 8]]

        sbFitsDict["S0"].append([sbNum, S0, d0])
        sbFitsDict["S1"].append([sbNum, S1, d1])
        sbFitsDict["S2"].append([sbNum, S2, d2])
        sbFitsDict["S3"].append([sbNum, S3, d3])

        # append alpha value
        thisData.append(np.arctan2(S2, S1) / 2 * 180. / np.pi)
        # append alpha error
        variance = (d2 ** 2 * S1 ** 2 + d1 ** 2 * S2 ** 2) / (S1 ** 2 + S2 ** 2) ** 2
        thisData.append(np.sqrt(variance) * 180. / np.pi)

        sbFitsDict["alpha"].append([sbNum, thisData[-2], thisData[-1]])

        # append gamma value
        thisData.append(np.arctan2(S3, np.sqrt(S1 ** 2 + S2 ** 2)) / 2 * 180. / np.pi)
        # append gamma error
        variance = (d3 ** 2 * (S1 ** 2 + S2 ** 2) ** 2 + (d1 ** 2 * S1 ** 2 + d2 ** 2 * S2 ** 2) * S3 ** 2) / (
        (S1 ** 2 + S2 ** 2) * (S1 ** 2 + S2 ** 2 + S3 ** 2) ** 2)
        thisData.append(np.sqrt(variance) * 180. / np.pi)
        sbFitsDict["gamma"].append([sbNum, thisData[-2], thisData[-1]])

        # append degree of polarization
        thisData.append(np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / S0)
        variance = ((d1 ** 2 * S0 ** 2 * S1 ** 2 + d0 ** 2 * (S1 ** 2 + S2 ** 2 + S3 ** 2) ** 2 + S0 ** 2 * (
        d2 ** 2 * S2 ** 2 + d3 ** 2 * S3 ** 2)) / (S0 ** 4 * (S1 ** 2 + S2 ** 2 + S3 ** 2)))
        thisData.append(np.sqrt(variance))
        sbFitsDict["DOP"].append([sbNum, thisData[-2], thisData[-1]])

        sbFits.append(thisData)

    sbFits = np.array(sbFits)
    sbFitsDict = {k: np.array(v) for k, v in sbFitsDict.items()}
    # This chunk used to insert the "alpha deviation", the difference between the angles and the
    # nir. I don't think I use this anymore, so stop saving it
                # origin_header = 'Sideband,S0,S0 err,S1,S1 err,S2,S2 err,S3,S3 err,alpha,alpha deviation,alpha err,gamma,gamma err,DOP,DOP err\n'
                # origin_header += 'Order,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,deg,deg,deg,deg,deg,arb.u.,arb.u.\n'
                # origin_header += 'Sideband,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(*["{}".format(series)] * 15)
                # sbFits = np.array(sbFits)
                # sbFits = np.insert(sbFits, 10, sbFits[:, 9] - lAlpha, axis=1)
                # sbFits = sbFits[sbFits[:, 0].argsort()]

    origin_header = "#\n"*100 # to fit all other files for easy origin importing
    origin_header += 'Sideband,S0,S0 err,S1,S1 err,S2,S2 err,S3,S3 err,alpha,alpha err,gamma,gamma err,DOP,DOP err\n'
    origin_header += 'Order,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,deg,deg,deg,deg,arb.u.,arb.u.\n'
    origin_header += 'Sideband,{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(*["{}".format(series)] * 14)
    sbFits = sbFits[sbFits[:, 0].argsort()]

    if isinstance(save, str):
        sbFitsSave = sbFits
        if not kwargs.get("saveStokes", True):
            headerlines = origin_header.splitlines()
            ln, units, coms = headerlines[-3:]
            ln = ','.join([ln.split(',')[0]] + ln.split(',')[9:])
            units = ','.join([units.split(',')[0]] + units.split(',')[9:])
            coms = ','.join([coms.split(',')[0]] + coms.split(',')[9:])
            headerlines[-3:] = ln, units, coms
            # remove them from the save data
            origin_header = '\n'.join(headerlines)
            sbFitsSave = np.delete(sbFits, range(1, 9), axis=1)

        if not os.path.exists(os.path.dirname(save)):
            os.mkdir(os.path.dirname(save))
        np.savetxt(save, np.array(sbFitsSave), delimiter=',', header=origin_header,
                   comments='', fmt='%.6e')

    # print("a = {:.2f} ± {:.2f}".format(sbFits[1, 9], sbFits[1, 10]))
    # print("g = {:.2f} ± {:.2f}".format(sbFits[1, 11], sbFits[1, 12]))

    if plot:
        plt.figure("alpha")
        plt.errorbar(sbFitsDict["alpha"][:, 0],
                     sbFitsDict["alpha"][:, 1],
                     sbFitsDict["alpha"][:, 2],
                     'o-', name = series
                     )
        plt.figure("gamma")
        plt.errorbar(sbFitsDict["gamma"][:, 0],
                     sbFitsDict["gamma"][:, 1],
                     sbFitsDict["gamma"][:, 2],
                     'o-', name=series
                     )
    return sbFits, sbFitsDict

####################
# Helper functions
####################
def fvb_crr(raw_array, offset=0, medianRatio=1, noiseCoeff=5, debugging=False):
    """

        Remove cosmic rays from a sequency of identical exposures
        :param raw_array: The array to be cleaned. Successive spectra should
                be the columns (i.e. 1600 x n) of the raw_array
        :param offset: baseline to add to raw_array.
               Not used, but here if it's needed in the future
        :param medianRatio: Multiplier to the median when deciding a cutoff
        :param noiseCoeff: Multiplier to the noise on the median
                    May need changing for noisy data
        :return:
    """

    d = np.array(raw_array)

    med = ndimage.filters.median_filter(d, size=(1, d.shape[1]), mode='wrap')
    med = np.median(d, axis=1).reshape(d.shape[0], 1)
    if debugging:
        print("shape of median filter:", med.shape)
    meanMedian = med.mean(axis=1)
    # meanMedian = med.copy()
    if debugging:
        print("shape of meaned median filter:", meanMedian.shape)
    # Construct a cutoff for each pixel. It was kind of guess and
    # check
    cutoff = meanMedian * medianRatio + noiseCoeff * np.std(meanMedian[-100:])
    if debugging:
        print("shape of cutoff criteria:", cutoff.shape)
        import pyqtgraph as pg

        winlist = []
        app = pg.QtGui.QApplication([])

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("Raw Image")
        p1 = win.addPlot()

        img = pg.ImageItem()
        img.setImage(d.copy().T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win.addItem(hist)

        win.nextRow()
        p2 = win.addPlot(colspan=2)
        p2.setMaximumHeight(250)
        p2.addLegend()
        for i, v in enumerate(d.T):
            p2.plot(v, pen=(i, d.shape[1]), name=str(i))
        p2.plot(np.sum(d, axis=1), pen=pg.mkPen('w', width=3))
        win.show()
        winlist.append(win)

        win2 = pg.GraphicsLayoutWidget()
        win2.setWindowTitle("Median Image")
        p1 = win2.addPlot()

        img = pg.ImageItem()
        img.setImage(med.T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win2.addItem(hist)

        win2.nextRow()
        p2 = win2.addPlot(colspan=2)
        p2.setMaximumHeight(250)

        p2.plot(np.sum(med, axis=1) / d.shape[1])
        win2.show()
        winlist.append(win2)

        win2 = pg.GraphicsLayoutWidget()
        win2.setWindowTitle("d-m")
        p1 = win2.addPlot()

        img = pg.ImageItem()
        img.setImage((d - med).T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win2.addItem(hist)

        win2.nextRow()
        p2 = win2.addPlot(colspan=2)
        p2.setMaximumHeight(250)
        p2.addLegend()
        for i, v in enumerate((d - med).T):
            p2.plot(v, pen=(i, d.shape[1]), name=str(i))
        p2.plot(cutoff, pen=pg.mkPen('w', width=3))
        win2.show()
        winlist.append(win2)

    # Find the bad pixel positions
    # Note the [:, None] - needed to cast the correct shapes
    badPixs = np.argwhere((d - med) > (cutoff.reshape(len(cutoff), 1)))

    for pix in badPixs:
        # get the other pixels in the row which aren't the cosmic
        if debugging:
            print("cleaning pixel", pix)
        p = d[pix[0], [i for i in range(d.shape[1]) if not i == pix[1]]]
        if debugging:
            print("\tRemaining pixels in row are", p)
        # Replace the cosmic by the average of the others
        # Could get hairy if more than one cosmic per row.
        # Maybe when doing many exposures?
        d[pix[0], pix[1]] = np.mean(p)

    if debugging:
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("Clean Image")
        p1 = win.addPlot()

        img = pg.ImageItem()
        img.setImage(d.copy().T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win.addItem(hist)

        win.nextRow()
        p2 = win.addPlot(colspan=2)
        p2.setMaximumHeight(250)
        p2.plot(np.sum(d, axis=1))
        win.show()
        winlist.append(win)
        app.exec_()

    return np.array(d)

def stitchData(dataList, plot=False):
    """
    Attempt to stitch together absorbance data. Will translate the second data set
    to minimize leastsq between the two data sets.
    :param dataList: Iterable of the data sets to be fit. Currently
            it only takes the first two elements of the list, but should be fairly
            straightforward to recursivly handle a list>2. Shifts the second
            data set to overlap the first

             elements of dataList can be either np.arrays or Absorbance class,
              where it will take the proc_data itself
    :param plot: bool whether or not you want the fit iterations to be plotted
            (for debugging)
    :return: a, a (2,) np.array of the shift
    """

    # Data coercsion, make sure we know what we're working wtih
    first = dataList[0]
    if isinstance(first, Absorbance):
        first = first.proc_data
        second = dataList[1]
    if isinstance(second, Absorbance):
        second = second.proc_data
    if plot:
        # Keep a reference to whatever plot is open at call-time
        # Useful if the calling script has plots before and after, as
        # omitting this will cause future plots to be added to figures here
        firstFig = plt.gcf()
        plt.figure("Stitcher")
        # Plot the raw input data
        plt.plot(*first.T)
        plt.plot(*second.T)

    # Algorithm is set up such that the "second" data set spans the
    # higher domain than first. Need to enforce this, and remember it
    # so the correct shift is applied
    flipped = False
    if max(first[:, 0]) > max(second[:, 0]):
        flipped = True
        first, second = second, first

def stitch_hsg_dicts(full_obj, new_obj, need_ratio=False, verbose=False, ratios=[1,1],
                     override_ratio = False, ignore_weaker_lowers = True):
    """
    This helper function takes a FullHighSideband and a sideband
    object, either CCD or PMT and smushes the new sb_results into the full_dict.

    The first input doesn't change, so f there's a PMT set of data involved, it
    should be in the full variable to keep the laser normalization intact.

    This function almost certainly does not work for stitching many negative orders
    in it's current state

    11/14/16
    --------
    This function has been updated to take the CCD objects themselves to be more
    intelligent about stitching. Consider two scans, (a) spec step 0 with 1 gain, spec
    step 2 with 110 gain and (b) spec step 0 with 50 gain and spec step 1 with 110 gain.
    The old version would always take spec step 0 to scale to, so while comparisons
    between spec step 0 and 1 for either case is valid, comparison between (a) and (b)
    were not, since they were scaled to different gain parameters. This new code will
    check what the gain values are and scale to the 110 data set, if present. This seems
    valid because we currently always have a 110 gain exposure for higher order
    sidebands.
    The exception is if the laser is present (sideband 0), as that is an absolute
    measure to which all else should be related.
    TODO: run some test cases to test this.

    06/11/18
    --------
    That sometimes was breaking if there were only 3-4 sidebands to fit with poor
    SNR. I've added the override_ratio to be passed to set a specific ratio to scale
    by. From data on 06/03/18, the 50gain to 110gain is a ~3.6 ratio. I haven't done
    a clean way of specifying which data set it should be scaled. Right now,
    it leaves the laser line data, or the 110 gain data alone.


    Inputs:
    full = full_dict from FullHighSideband, or HighSidebandPMT.  It's important
           that it contains lower orders than the new_dict.
    new_dict = another full_dict.
    need_ratio = If gain or other parameters aren't equal and must resort to
                 calculating the ratio instead of the measurements being equivalent.
                 Changing integration time still means N photons made M counts,
                 but changing gain or using PMT or whatever does affect things.
    ratios: Will update with the values to the ratios needed to scale the data.
            ratios[0] is the ratio for the "full_obj"
            ratios[1] is the ratio for the "new_obj"
            one of them will be one, one will be the appropriate scale, since one of
            them is unscaled. This is strictly speaking an output
    override_ratio: Pass a float to specify the ratio that should be used.
    ignore_weaker_lowers: Sometimes, a SB is in the short pass filter so a lower
        order is weaker than the next highest. If True, causes script to ignore all
        sidebands which are weaker and lower order.

    Returns:
    full = extended version of the input full.  Overlapping sidebands are
           averaged because that makes sense?
    """
    if isinstance(full_obj, dict) and isinstance(new_obj, dict):
        return stitch_hsg_dicts_old(full_obj, new_obj, need_ratio, verbose)

    if verbose:
        print("=" * 15)
        print()
        print("Stitching HSG dicts")
        print()
        print("=" * 15)

    # remove potentially offensive SBs, i.e. a 6th order SB being in the SPF for more
    #  data, but being meaningless to pull intensity information from.
    # Note: this might not be the best if you get to higher order stitches where it's
    #  possible that the sidebands might not be monotonic (from noise?)
    if ignore_weaker_lowers:
        full_obj.full_dict, full_obj.sb_results = FullHighSideband.parse_sb_array(full_obj.sb_results)
        new_obj.new_dict, new_obj.sb_results = FullHighSideband.parse_sb_array(new_obj.sb_results)

    # was fucking around with references and causing updates to arrays when it shouldn't
    # be
    full = copy.deepcopy(full_obj.full_dict)
    new_dict = copy.deepcopy(new_obj.full_dict)

    # Force a rescaling if you've passed a specified parameter
    # if isinstance(override_ratio, float):
    #     need_ratio = True

    # Do some testing to see which dict should be scaled to the other
    # I honestly forget why I prioritized the PMT first like this. But the third
    # check looks to make a gain 110 prioritize non-110, unless the non-110 includes
    # a laser line
    scaleTo = ""
    if need_ratio:
        if isinstance(new_obj, HighSidebandPMT):
            scaleTo = "new"
        elif isinstance(full_obj, HighSidebandPMT):
            scaleTo = "full"
        elif new_obj.parameters["gain"] == 110 and full_obj.parameters["gain"] != 110 \
            and 0 not in full:
            scaleTo = "new"
        else:
            scaleTo = "full"

    if verbose:
        print("\tI'm adding these sidebands", sorted(new_dict.keys()))
        print("\t  With these:", sorted(full.keys()))
    overlap = [] # The list that hold which orders are in both dictionaries
    missing = [] # How to deal with sidebands that are missing from full but in new.
    for new_sb in sorted(new_dict.keys()):
        full_sbs = sorted(full.keys())
        if new_sb in full_sbs:
            overlap.append(new_sb)
        elif new_sb not in full_sbs and new_sb < full_sbs[-1]:
            # This probably doesn't work with bunches of negative orders
            missing.append(new_sb)

    if verbose:
        print("\t  ( overlap:", overlap, ")")
        print("\t  ( missing:", missing, ")")


    # This if-else clause handles how to average together overlapping sidebands
    # which are seen in both spectra,
    if need_ratio:
        # Calculate the appropriate ratio to multiply the new sidebands by.
        # I'm not entirely sure what to do with the error of this guy.
        ratio_list = []
        try:
            new_starter = overlap[-1]
            if verbose:
                print("\n\tadding these ratios,", end=' ')
            if len(overlap) > 2:
                overlap = [x for x in overlap if (x % 2 == 0)
                           ]# and (x != min(overlap) and (x != max(overlap)))]
            if scaleTo == "new":
                if verbose:
                    print("scaling to new :")
                for sb in overlap:
                    ratio_list.append(new_dict[sb][2]/full[sb][2])
                    if verbose:
                        print("\t\t{:2.0f}: {:.3e}/{:.3e} ~ {:.3e},".format(sb, new_dict[sb][2],
                                                               full[sb][2], ratio_list[-1]))
                # new_ratio = 1 06/11/18 Not sure what these were used for
                ratio = np.mean(ratio_list)
            else:
                if verbose:
                    print("scaling to full:")
                for sb in overlap:
                    ratio_list.append(full[sb][2] / new_dict[sb][2])
                    if verbose:
                        print("\t\t{:2.0f}: {:.3e}/{:.3e} ~ {:.3e},".format(sb, full[sb][2],
                                                               new_dict[sb][2], ratio_list[-1]))
                # new_ratio = np.mean(ratio_list) 06/11/18 Not sure what these were used for

                ratio = np.mean(ratio_list)
            # Maybe not the best way to do it, performance wise, since you still
            # iterate through the list, even though you'll override it.
            if isinstance(override_ratio, float):
                ratio = override_ratio
                if verbose:
                    print("overriding calculated ratio with user inputted")
            error = np.std(ratio_list) / np.sqrt(len(ratio_list))

        except IndexError:
            # If there's no overlap (which you shouldn't let happen), hardcode a ratio
            # and error. I looked at all the ratios for the overlaps from 6/15/16
            # (540ghz para) to get the rough average. Hopefully they hold for all data.
            if not overlap:
                ratio = 0.1695
                error = 0.02
                # no overlap, so make sure it grabs all the sidebands
                new_starter = min(new_dict.keys())
            else:
                raise
        if verbose:
            # print "Ratio list\n\t", ("{:.3g}, "*len(ratio_list))[:-2].format(*ratio_list)
            # print "Overlap   \n\t", [round(ii, 3) for ii in overlap]
            print("\t Ratio: {:.3g} +- {:.3g} ({:.2f}%)\n".format(ratio, error, error/ratio*100))
        # Adding the new sidebands to the full set and moving errors around.
        # I don't know exactly what to do about the other aspects of the sidebands
        # besides the strength and its error.
        if scaleTo == "full":
            ratios[1] = ratio
            for sb in overlap:
                if verbose:
                    print("For SB {:02d}, original strength is {:.3g} +- {:.3g} ({:.3f}%)".format(int(sb), new_dict[sb][2], new_dict[sb][3],
                                                                        new_dict[sb][3]/new_dict[sb][2]*100
                            ))

                new_dict[sb][3] = ratio * new_dict[sb][2] * np.sqrt((error / ratio) ** 2 + (new_dict[sb][3] / new_dict[sb][2]) ** 2)
                new_dict[sb][2] = ratio * new_dict[sb][2]
                if verbose:
                    print("\t\t   scaled\t\t\t\t{:.3g} +- {:.3g} ({:.3f}%)".format(new_dict[sb][2],
                                                                        new_dict[sb][3],
                                                                        new_dict[sb][3]/new_dict[sb][2]*100))
                    print("\t\t   full\t\t\t\t\t{:.3g} +- {:.3g} ({:.3f}%)".format(full[sb][2],
                                                                        full[sb][3],
                                                                        full[sb][3]/full[sb][2]*100))


                sb_error = np.sqrt(full[sb][3] ** (-2) + new_dict[sb][3] ** (-2)) ** (-1)

                avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (
                    new_dict[sb][3] ** 2)) / (full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                full[sb][2] = avg
                full[sb][3] = sb_error
                if verbose:
                    print("\t\t   replaced with \t\t{:.3g} +- {:.3g} ({:.3f}%)".format(full[sb][2],
                                                                        full[sb][3],
                                                                        full[sb][3]/full[sb][2]*100))
                    print()

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (
                new_dict[sb][5] ** 2)) / (
                             full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error
        else:
            ratios[0] = ratio
            for sb in overlap:
                full[sb][3] = ratio * full[sb][2] * np.sqrt((error / ratio) ** 2 + (full[sb][3] / full[sb][2]) ** 2)
                full[sb][2] = ratio * full[sb][2]

                sberror = np.sqrt(full[sb][3] ** (-2) + new_dict[sb][3] ** (-2)) ** (-1)
                avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (
                    new_dict[sb][3] ** 2)) / (full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                full[sb][2] = avg
                full[sb][3] = sberror

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (
                new_dict[sb][5] ** 2)) / (
                             full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error


    else: # not needing a new ratio
        try:
            new_starter = overlap[-1] # This grabs the sideband order where only the new dictionary has
                                      # sideband information.  It's not clear why it necessarily has to be
                                      # at this line.
            overlap = [x for x in overlap if (x % 2 == 0)
                       ] # and (x != min(overlap) and (x != max(overlap)))]
            # This cuts out the lowest order sideband in the overlap for mysterious reasons
            for sb in overlap: # This for loop average two data points weighted by their relative errors
                if verbose:
                    print("The sideband", sb)
                    print("Old value", full[sb][4] * 1000)
                    print("Add value", new_dict[sb][4] * 1000)
                try:
                    error = np.sqrt(full[sb][3] ** (-2) + new_dict[sb][3] ** (-2)) ** (-1)
                    avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (new_dict[sb][3] ** 2)) / (
                        full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                    full[sb][2] = avg
                    full[sb][3] = error
                except RuntimeWarning:
                    raise IOError()

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (new_dict[sb][5] ** 2)) / (
                full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error
                if verbose:
                    print("New value", lw_avg * 1000)
        except:
            new_starter = 0  # I think this makes things work when there's no overlap
    if verbose:
        print("appending new elements. new_starter={}".format(new_starter))


    for sb in [x for x in list(new_dict.keys()) if ((x > new_starter) or (x in missing))]:
        full[sb] = new_dict[sb]
        if scaleTo == "full":
            full[sb][2] = ratio * full[sb][2]
            full[sb][3] = full[sb][2] * np.sqrt((error / ratio) ** 2 + (ratio * full[sb][3] / full[sb][2]) ** 2)
    if scaleTo == "new":
        for sb in set(full.keys()) - set(sorted(new_dict.keys())[:]):
            full[sb][2] *= ratio
            # TODO: I think this is an invalid error
            # propagation (since ratio has error associated with it
            full[sb][3] *= ratio
    if verbose:
        print("I made this dictionary", sorted(full.keys()))
        print('-'*19)
        return full
        return full, ratio #the fuck? Why was this here?

    return full

def stitch_hsg_dicts_old(full, new_dict, need_ratio=False, verbose=False):
    """
    This helper function takes a FullHighSideband.full_dict attribute and a sideband
    object, either CCD or PMT and smushes the new sb_results into the full_dict.

    The first input doesn't change, so f there's a PMT set of data involved, it
    should be in the full variable to keep the laser normalization intact.

    This function almost certainly does not work for stitching many negative orders
    in it's current state

    11/14/16
    --------
    The original function has been updated to take the full object (instead of
    the dicts alone) to better handle calculating ratios when stitching. This is called
    once things have been parsed in the original function (or legacy code where dicts
    are passed instead of the object)

    Inputs:
    full = full_dict from FullHighSideband, or HighSidebandPMT.  It's important
           that it contains lower orders than the new_dict.
    new_dict = another full_dict.
    need_ratio = If gain or other parameters aren't equal and must resort to
                 calculating the ratio instead of the measurements being equivalent.
                 Changing integration time still means N photons made M counts,
                 but changing gain or using PMT or whatever does affect things.

    Returns:
    full = extended version of the input full.  Overlapping sidebands are
           averaged because that makes sense?
    """
    if verbose:
        print("I'm adding these sidebands in old stitcher", sorted(new_dict.keys()))
    overlap = [] # The list that hold which orders are in both dictionaries
    missing = [] # How to deal with sidebands that are missing from full but in new.
    for new_sb in sorted(new_dict.keys()):
        full_sbs = sorted(full.keys())
        if new_sb in full_sbs:
            overlap.append(new_sb)
        elif new_sb not in full_sbs and new_sb < full_sbs[-1]:
            # This probably doesn't work with bunches of negative orders
            missing.append(new_sb)

    if verbose:
        print("overlap:", overlap)
        print("missing:", missing)

    # This if-else clause handles how to average together overlapping sidebands
    # which are seen in both spectra,
    if need_ratio:
        # Calculate the appropriate ratio to multiply the new sidebands by.
        # I'm not entirely sure what to do with the error of this guy.
        ratio_list = []
        #print '\n1979\nfull[2]', full[0][2]
        try:
            new_starter = overlap[-1]
            if len(overlap) > 2:
                overlap = [x for x in overlap if (x % 2 == 0)
                           ]#and (x != min(overlap) and (x != max(overlap)))]
            for sb in overlap:
                ratio_list.append(full[sb][2] / new_dict[sb][2])
            ratio = np.mean(ratio_list)
            # print
            # print '-'*15
            # print "ratio for {}: {}".format()
            error = np.std(ratio_list) / np.sqrt(len(ratio_list))
        except IndexError:
            # If there's no overlap (which you shouldn't let happen),
            # hardcode a ratio and error.
            # I looked at all the ratios for the overlaps from 6/15/16
            # (540ghz para) to get the rough average. Hopefully they hold
            # for all data.
            if not overlap:
                ratio = 0.1695
                error = 0.02
                # no overlap, so make sure it grabs
                # all the sidebands
                new_starter = min(new_dict.keys())
            else:
                raise
        if verbose:
            print("Ratio list","\n", [round(ii, 3) for ii in ratio_list])
            print("Overlap   ","\n", [round(ii, 3) for ii in overlap])
            print("Ratio", ratio)
            print("Error", error)
        #print '\n2118\nfull[2]', full[0][2]
        # Adding the new sidebands to the full set and moving errors around.
        # I don't know exactly what to do about the other aspects of the sidebands
        # besides the strength and its error.
        for sb in overlap:
            full[sb][2] = ratio * new_dict[sb][2]
            full[sb][3] = full[sb][2] * np.sqrt((error / ratio) ** 2 + (new_dict[sb][3] / new_dict[sb][2]) ** 2)
            #print '\n2125\nfull[2]', full[0][3]
            # Now for linewidths
            lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
            lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (new_dict[sb][5] ** 2)) / (
            full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
            full[sb][4] = lw_avg
            full[sb][5] = lw_error
        #print '\n2132\nfull[2]', full[0][2]
    else:
        try:
            new_starter = overlap[-1] # This grabs the sideband order where only the new dictionary has
                                      # sideband information.  It's not clear why it necessarily has to be
                                      # at this line.
            overlap = [x for x in overlap if (x % 2 == 0) and (x != min(overlap) and (x != max(overlap)))]
            # This cuts out the lowest order sideband in the overlap for mysterious reasons
            for sb in overlap: # This for loop average two data points weighted by their relative errors
                if verbose:
                    print("The sideband", sb)
                    print("Old value", full[sb][4] * 1000)
                    print("Add value", new_dict[sb][4] * 1000)
                error = np.sqrt(full[sb][3] ** (-2) + new_dict[sb][3] ** (-2)) ** (-1)
                avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (new_dict[sb][3] ** 2)) / (
                    full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                full[sb][2] = avg
                full[sb][3] = error

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (new_dict[sb][5] ** 2)) / (
                full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error
                if verbose:
                    print("New value", lw_avg * 1000)
        except:

            new_starter = 0  # I think this makes things work when there's no overlap
    if verbose:
        print("appending new elements. new_starter={}".format(new_starter))

    # This loop will add the sidebands which were only seen in the second step
    for sb in [x for x in list(new_dict.keys()) if ((x >= new_starter) or (x in missing))]:
        full[sb] = new_dict[sb]
        if need_ratio:
            full[sb][2] = ratio * full[sb][2]
            full[sb][3] = full[sb][2] * np.sqrt((error / ratio) ** 2 + (ratio * full[sb][3] / full[sb][2]) ** 2)
            #print '\n2164\nfull[2]', full[0][2]
    if verbose:
        print("I made this dictionary", sorted(full.keys()))
    return full

def save_parameter_sweep_no_sb(spectrum_list, file_name, folder_str, param_name, unit,
                         verbose=False):
    """
    This function will take a fully processed list of spectrum objects and
    slice Spectrum.sb_fits appropriately to get an output like:

    "Parameter" | SB1 freq | err | SB1 amp | error | SB1 linewidth | error | SB2...| SBn...|
    param1      |    .     |
    param2      |    .     |
      .
      .
      .

    Currently I'm thinking fuck the offset y0
    After constructing this large matrix, it will save it somewhere.
    """
    spectrum_list.sort(key=lambda x: x.parameters[param_name])
    included_spectra = dict()
    param_array = None
    sb_included = []

    for spec in spectrum_list:
        sb_included = sorted(list(set(sb_included + list(spec.full_dict.keys()))))
        included_spectra[spec.fname.split('/')[-1]] = spec.parameters[param_name]
        # If these are from summed spectra, then only the the first file name
        # from that sum will show up here, which should be fine?
    if verbose:
        # print "full name:", spectrum_list[0].fname
        print("included names:", included_spectra)
        print("sb_included:", sb_included)

    for spec in spectrum_list:
        temp_dict = {}  # This is different from full_dict in that the list has the
        # sideband order as the zeroth element.
        if verbose:
            print("the sb_results:", spec.sb_results)
        if spec.sb_results.ndim == 1: continue
        for index in range(len(spec.sb_results[:, 0])):
            if verbose:
                print("my array slice:", spec.sb_results[index, :])
            temp_dict[int(round(spec.sb_results[index, 0]))] = np.array(
                spec.sb_results[index, 1:])

        if verbose:
            print(temp_dict)

        for sb in sb_included:
            blank = np.zeros(6)
            # print "checking sideband order:", sb
            # print "blank", blank
            if sb not in temp_dict:
                # print "\nNeed to add sideband order:", sb
                temp_dict[sb] = blank
        try:  # Why is this try-except here?
            spec_data = np.array([float(spec.parameters[param_name])])
        except:
            spec_data = np.array([float(spec.parameters[param_name][:2])])
        for key in sorted(temp_dict.keys()):
            # print "I am going to hstack this:", temp_dict[key]
            spec_data = np.hstack((spec_data, temp_dict[key]))

        try:
            param_array = np.vstack((param_array, spec_data))
        except:
            param_array = np.array(spec_data)
        if verbose:
            print("The shape of the param_array is:", param_array.shape)
            # print "The param_array itself is:", param_array
    '''
    param_array_norm = np.array(param_array).T # python iterates over rows
    for elem in [x for x in xrange(len(param_array_norm)) if (x-1)%7 == 3]:
        temp_max = np.max(param_array_norm[elem])
        param_array_norm[elem] = param_array_norm[elem] / temp_max
        param_array_norm[elem + 1] = param_array_norm[elem + 1] / temp_max
    '''
    snipped_array = param_array[:, 0]
    norm_array = param_array[:, 0]
    if verbose:
        print("Snipped_array is", snipped_array)
    for ii in range(len(param_array.T)):
        if (ii - 1) % 6 == 0:
            if verbose:
                print("param_array shape", param_array[:, ii])
            snipped_array = np.vstack((snipped_array, param_array[:, ii]))
            norm_array = np.vstack((norm_array, param_array[:, ii]))
        elif (ii - 1) % 6 == 2:
            snipped_array = np.vstack((snipped_array, param_array[:, ii]))

            temp_max = np.max(param_array[:, ii])
            norm_array = np.vstack((norm_array, param_array[:, ii] / temp_max))
        elif (ii - 1) % 6 == 3:
            snipped_array = np.vstack((snipped_array, param_array[:, ii]))
            norm_array = np.vstack((norm_array, param_array[:, ii] / temp_max))

    snipped_array = snipped_array.T
    norm_array = norm_array.T

    try:
        os.mkdir(folder_str)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    norm_name = file_name + '_norm.txt'
    snip_name = file_name + '_snip.txt'
    file_name = file_name + '.txt'

    try:
        included_spectra_str = json.dumps(included_spectra, sort_keys=True, indent=4,
                                          separators=(',', ': '))
    except:
        print("Source: save_parameter_sweep\nJSON FAILED")
        return
    included_spectra_str = included_spectra_str.replace('\n', '\n#')

    included_spectra_str += '\n#' * (99 - included_spectra_str.count('\n'))
    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    for order in sb_included:
        origin_import1 += "Frequency,error,Sideband strength,error,Linewidth,error"
        origin_import2 += ",eV,,arb. u.,,meV,"
        origin_import3 += ",{0},,{0},,{0},".format(order)
    origin_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    for order in sb_included:
        origin_import1 += ",Frequency,Sideband strength,error"
        origin_import2 += ",eV,arb. u.,"
        origin_import3 += ",{0},{0},".format(order)
    origin_snip = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    header_total = '#' + included_spectra_str + '\n' + origin_total
    header_snip = '#' + included_spectra_str + '\n' + origin_snip

    # print "Spec header: ", spec_header
    if verbose:
        print("the param_array is:", param_array)
    np.savetxt(os.path.join(folder_str, file_name), param_array, delimiter=',',
               header=header_total, comments='', fmt='%0.6e')
    np.savetxt(os.path.join(folder_str, snip_name), snipped_array, delimiter=',',
               header=header_snip, comments='', fmt='%0.6e')
    np.savetxt(os.path.join(folder_str, norm_name), norm_array, delimiter=',',
               header=header_snip, comments='', fmt='%0.6e')
    if verbose:
        print("Saved the file.\nDirectory: {}".format(
            os.path.join(folder_str, file_name)))

def save_parameter_sweep(spectrum_list, file_name, folder_str, param_name, unit,
                         wanted_indices = [1, 3, 4], skip_empties = False, verbose=False,
                         header_dict = {}, only_even=False):
    """
    This function will take a fully processed list of spectrum objects and
    slice Spectrum.sb_fits appropriately to get an output like:

    "Parameter" | SB1 freq | err | SB1 amp | error | SB1 linewidth | error | SB2...| SBn...|
    param1      |    .     |
    param2      |    .     |
      .
      .
      .

    Currently I'm thinking fuck the offset y0
    After constructing this large matrix, it will save it somewhere.


    Thus function has been update to pass a list of indices to slice for the return
    values

    skip_empties: If False, will add a row of zeroes for the parameter even if no sidebands
    are found. If True, will not add a line for that parameter

    only_even: don't include odd orders in the saved sweep

    [sb number, Freq (eV), Freq error (eV), Gauss area (arb.), Area error, Gauss linewidth (eV), Linewidth error (eV)]
    [    0    ,      1   ,        2,      ,        3         ,      4    ,         5           ,        6            ]
    """
    if isinstance(param_name, list):
        # if you pass two things because the param you want
        # is in a dict (e.g. field strength has mean/std)
        # do it that way
        param_name_list = list(param_name) # keep reference to old one
        paramGetter = lambda x: x.parameters[param_name_list[0]][param_name_list[1]]
        # Keep the name for labeling things later on
        param_name = param_name[0]
    else:
        paramGetter = lambda x: x.parameters[param_name]

    # Sort all of the spectra based on the desired key
    spectrum_list.sort(key=paramGetter)

    # keep track of which file name corresponds to which parameter which gets put in
    included_spectra = dict()

    # The big array which will be stacked up to keep all of the sideband details vs desired parameter
    param_array = None

    # list of which sidebands are seen throughout.
    sb_included = []
    # how many parameters (area, strength, linewidth, pos, etc.) are there?
    # Here incase software changes and more things are kept in
    # sb results. Needed to handle how to slice the arrays
    try:
        num_params = spectrum_list[0].sb_results.shape[1]
    except IndexError:
        # There's a file with only 1 sb and it happens to be first
        # in the list.
        num_params = spectrum_list[0].sb_results.shape[0]
    except AttributeError:
        # The first file has no sidebands, so just hardcode it, as stated below.
        num_params=0

    # Rarely, there's an issue where I'm doing some testing and there's a set
    # where the first file has no sidebands in it, so the above thing returns 0
    # It seems really silly to do a bunch of testing to try and correct for that, so
    # I'm going to hardcode the number of parameters.
    if num_params == 0:
        num_params = 7

    # loop through all of them once to figure out which sidebands are seen in all spectra
    for spec in spectrum_list:
        try:
            # use sets to keep track of only unique sidebands
            sb_included = sorted(list(set(sb_included + list(spec.full_dict.keys()))))
        except AttributeError:
            print("No full dict?", spec.fname)
            print(spec.sb_list)
        # If these are from summed spectra, then only the the first file name
        # from that sum will show up here, which should be fine?
        included_spectra[spec.fname.split('/')[-1]] = paramGetter(spec)

    if only_even:
        sb_included = [ii for ii in sb_included if not ii%2]
    if verbose:
        print("included names:", included_spectra)
        print("sb_included:", sb_included)

    for spec in spectrum_list:
        # Flag to keep whethere there are no sidebands or not. Used to skip
        # issues when trying to index on empty arrays
        noSidebands = False
        if verbose:
            print("the sb_results:", spec.sb_results)

        # if no sidebands were found, skip this one
        try:
            # TODO: (08/14/18) the .ndim==1 isn't the correct check, since it fails
            # when looking at the laser line. Need to test this with a real
            # empty data set, vs data set with 1 sb
            #
            #
            # (08/28/18) I'm not sure what the "not spec" is trying to handle
            #      spec.sb_results is None occurs when _no_ sidebands were fit
            #     spec.sb_results.ndim == 1 happens when only one sideband is found
            if not spec or spec.sb_results is None or spec.sb_results.ndim == 1:
                if spec.sb_results is None:
                    # Flag no sidebands are afound
                    noSidebands = True
                elif spec.sb_results[0] == 0:
                    # Cast it to 2d to allow slicing later on. Not sure hwy this is
                    # only done if the laser line is the one found.
                    spec.sb_results = np.atleast_2d(spec.sb_results)
                elif skip_empties:
                    continue
                else:
                    noSidebands = True
        except (AttributeError, TypeError):
            # continue
            raise

        # Make an sb_results of all zeroes where we'll fill
        # in the sideband info we found
        new_spec = np.zeros((len(sb_included), num_params))
        if not noSidebands:
            sb_results = spec.sb_results.copy()
            saw_sbs = sb_results[:, 0]
            found_sb = sorted(list(set(sb_included) & set(saw_sbs)))
            found_idx = [sb_included.index(ii) for ii in found_sb]
            try:
                new_spec[:, 0] = sb_included
            except:
                print("new_spec", new_spec)
                raise
            try:
                if only_even:
                    new_spec[found_idx, :] = sb_results[sb_results[:,0]%2==0]
                else:
                    new_spec[found_idx, :] = sb_results
            except ValueError:
                print(spec.fname)
                print("included:", sb_included)
                print("found:", found_sb, found_idx)
                print(new_spec.shape, sb_results.shape)
                print(sb_results)
                print(new_spec)
                raise

        spec_data = np.insert(new_spec.flatten(), 0, float(paramGetter(spec)))

        try:
            param_array = np.row_stack((param_array, spec_data))
        except:
            param_array = np.array(spec_data)

    if param_array.ndim == 1: # if you only pass one spectra
        param_array = param_array[None, :] # recast it to 2D for slicing
    # the indices we want from the param array from the passed argument
    snip = wanted_indices
    N = len(sb_included)
    # run it out across all of the points across the param_array
    snipped_indices = [0] + list(
        1+np.array(snip * N) + num_params * np.array(sorted(list(range(N)) * len(snip))))
    snipped_array = param_array[:, snipped_indices]
    norm_array = snipped_array.copy()
    # normalize the area if it's requested
    if 3 in snip:
        num_snip = len(snip)
        strength_idx = snip.index(3)
        if 4 in snip:
            #normalize error first if it was requested
            idx = snip.index(4)
            norm_array[:, 1 + idx + np.arange(N) * num_snip] /= norm_array[:,1 + strength_idx + np.arange(N) * num_snip].max(axis=0)
        strength_idx = snip.index(3)
        norm_array[:, 1+strength_idx+np.arange(N)*num_snip]/=norm_array[:, 1+strength_idx+np.arange(N)*num_snip].max(axis=0)

    try:
        os.mkdir(folder_str)
    except TypeError:
        pass # if you pass None as folder_str (for using byteIO)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    included_spectra.update(header_dict)
    try:
        included_spectra_str = json.dumps(included_spectra, sort_keys=True, indent=4,
                                          separators=(',', ': '))
    except:
        print("Source: save_parameter_sweep\nJSON FAILED")
        return
    included_spectra_str = included_spectra_str.replace('\n', '\n#')

    included_spectra_str += '\n#' * (99 - included_spectra_str.count('\n'))

    # this will make the header chunk for the full, un-sliced data set
    # TODO: fix naming so you aren't looping twice
    ### 1/9/18 This isn't needed, right? Why isn't it deleted?
    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    for order in sb_included:
        origin_import1 += ",sideband,Frequency,error,Sideband strength,error,Linewidth,error"
        origin_import2 += ",order,eV,eV,arb. u.,arb.u.,meV,meV"
        origin_import3 += ",,{0},,{0},,{0},".format(order)
    origin_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3


    # This little chunk will make a chunk block of header strings for the sliced
    # data set which can be looped over
    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    wanted_titles = ["Sideband", "Frequency", "error", "Sideband strength","error","Linewidth","error"]
    wanted_units  = ["order", "eV", "eV", "arb. u.", "arb. u.", "eV", "eV"]
    wanted_comments = ["", "{0}", "", "{0}", "", "{0}", ""]
    wanted_titles = ",".join([wanted_titles[ii] for ii in wanted_indices])
    wanted_units = ",".join([wanted_units[ii] for ii in wanted_indices])
    wanted_comments = ",".join([wanted_comments[ii] for ii in wanted_indices])

    for order in sb_included:
        origin_import1 += ","+wanted_titles
        origin_import2 += ","+wanted_units
        origin_import3 += ","+wanted_comments.format(order)
    origin_snip = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    header_total = '#' + included_spectra_str + '\n' + origin_total
    header_snip = '#' + included_spectra_str + '\n' + origin_snip

    # print "Spec header: ", spec_header
    if verbose:
        print("the param_array is:", param_array)
    if isinstance(file_name, list):
        if isinstance(file_name[0], io.BytesIO):
            np.savetxt(file_name[0], param_array, delimiter=',',
                       header=header_total, comments='', fmt='%0.6e')
            np.savetxt(file_name[1], snipped_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            np.savetxt(file_name[2], norm_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            # Need to reset the file position if you want to read them immediately
            # Is it better to do that here, or assume you'll do it later?
            # I'm gonna assume here, because I can't currently think of a time when I'd want
            # to be at the end of the file
            [ii.seek(0) for ii in file_name]
            if verbose:
                print("Saved the file to bytes objects")
    else:
        if file_name:
            norm_name = file_name + '_norm.txt'
            snip_name = file_name + '_snip.txt'
            file_name = file_name + '.txt'
            np.savetxt(os.path.join(folder_str, file_name), param_array, delimiter=',',
                       header=header_total, comments='', fmt='%0.6e')
            np.savetxt(os.path.join(folder_str, snip_name), snipped_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            np.savetxt(os.path.join(folder_str, norm_name), norm_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            if verbose:
                print("Saved the file.\nDirectory: {}".format(os.path.join(folder_str, file_name)))
        else:
            if verbose:
                print("Didn't save")

    return sb_included, param_array, snipped_array, norm_array

def save_parameter_sweep_vs_sideband(spectrum_list, file_name,
                                     folder_str, param_name, unit, verbose=False,
                                     wanted_indices = [1, 3, 4]):
    """
    Similar to save_parameter_sweep, but the data[:,0] column is sideband number instead of
    series, and each set of columns correspond to a series step. Pretty much compiles
    all of the fit parameters from the files that are already saved and puts it into
    one file to keep from polluting the Origin folder
    :param spectrum_list:
    :param file_name:
    :param folder_str:
    :param param_name:
    :param unit:
    :param verbose:

    sb number is automatically prepended, so do not include in slicing list

    [sb number, Freq (eV), Freq error (eV), Gauss area (arb.), Area error, Gauss linewidth (eV), Linewidth error (eV)]
    [    0    ,      1   ,        2,      ,        3         ,      4    ,         5           ,        6            ]

    :return:
    """
    spectrum_list.sort(key=lambda x: x.parameters[param_name])
    included_spectra = dict()
    param_array = None
    sb_included = []

    # what parameters were included (for headers)
    params = sorted([x.parameters[param_name] for x in spectrum_list])

    for spec in spectrum_list:
        sb_included = sorted(list(set(sb_included + list(spec.full_dict.keys()))))
        included_spectra[spec.fname.split('/')[-1]] = spec.parameters[param_name]
        # If these are from summed spectra, then only the the first file name
        # from that sum will show up here, which should be fine?
    if verbose:
        # print "full name:", spectrum_list[0].fname
        print("included names:", included_spectra)
        print("sb_included:", sb_included)

    param_array = np.array(sb_included)

    for spec in spectrum_list:
        temp_dict = spec.full_dict.copy()

        #prevent breaking if no sidebands in spectrum
        if not temp_dict:
            if verbose:
                print("No sidebands here? {}, {}".format(spec.parameters["series"],
                                                         spec.parameters["spec_step"]))
            continue

        if verbose:
            print(temp_dict)

        # matrix for holding all of the sb information
        # for a given spectrum
        spec_matrix = None
        for sb in sb_included:
            blank = np.zeros(6)
            # print "checking sideband order:", sb
            # print "blank", blank
            sb_data = temp_dict.get(sb, blank)
            try:
                spec_matrix = np.row_stack((spec_matrix, sb_data))
            except:
                spec_matrix = sb_data
        param_array = np.column_stack((param_array, spec_matrix))

    # the indices we want from the param array
    # 1- freq, 3-area, 4-area error
    snip = wanted_indices
    N = len(spectrum_list)
    # run it out across all of the points across the param_array
    snipped_indices = [0] + list( np.array(snip*N) + 6*np.array(sorted(list(range(N))*len(snip))) )
    snipped_array = param_array[:, snipped_indices]

    try:
        os.mkdir(folder_str)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    snip_name = file_name + '_snip.txt'
    file_name = file_name + '.txt'

    try:
        included_spectra_str = json.dumps(included_spectra, sort_keys=True, indent=4, separators=(',', ': '))
    except:
        print("Source: save_parameter_sweep\nJSON FAILED")
        return
    included_spectra_str = included_spectra_str.replace('\n', '\n#')

    included_spectra_str += '\n#' * (99 - included_spectra_str.count('\n'))
    origin_import1 = "Sideband"
    origin_import2 = "Order"
    origin_import3 = "SB"
    for param in params:
        origin_import1 += ",Frequency,error,Sideband strength,error,Linewidth,error"
        origin_import2 += ",eV,,arb. u.,,meV,"
        origin_import3 += ",{0},,{0},,{0},".format(param)
    origin_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    # This little chunk will make a chunk block of header strings for the sliced
    # data set which can be looped over
    origin_import1 = "Sideband"
    origin_import2 = "Order"
    origin_import3 = "SB"
    wanted_titles = ["Sideband", "Frequency", "error", "Sideband strength", "error",
                     "Linewidth", "error"]
    wanted_units = ["order", "eV", "eV", "arb. u.", "arb. u.", "eV", "eV"]
    wanted_comments = ["", "{0}", "", "{0}", "", "{0}", ""]
    wanted_titles = ",".join([wanted_titles[ii] for ii in wanted_indices])
    wanted_units = ",".join([wanted_units[ii] for ii in wanted_indices])
    wanted_comments = ",".join([wanted_comments[ii] for ii in wanted_indices])

    for param in params:
        origin_import1 += "," + wanted_titles
        origin_import2 += "," + wanted_units
        origin_import3 += "," + wanted_comments.format(param)
    origin_snip = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    header_total = '#' + included_spectra_str + '\n' + origin_total
    header_snip = '#' + included_spectra_str + '\n' + origin_snip

    # print "Spec header: ", spec_header
    if verbose:
        print("the param_array is:", param_array)
    if file_name: # allow passing false (or empty string) to prevent saving
        np.savetxt(os.path.join(folder_str, file_name), param_array, delimiter=',',
                   header=header_total, comments='', fmt='%0.6e')
        np.savetxt(os.path.join(folder_str, snip_name), snipped_array, delimiter=',',
                   header=header_snip, comments='', fmt='%0.6e')
    if verbose:
        print("Saved the file.\nDirectory: {}".format(os.path.join(folder_str, file_name)))
    return None

def stitchData(dataList, plot=False):
    """
    Attempt to stitch together absorbance data. Will translate the second data set
    to minimize leastsq between the two data sets.
    :param dataList: Iterable of the data sets to be fit. Currently
            it only takes the first two elements of the list, but should be fairly
            straightforward to recursivly handle a list>2. Shifts the second
            data set to overlap the first
             elements of dataList can be either np.arrays or Absorbance class,
              where it will take the proc_data itself
    :param plot: bool whether or not you want the fit iterations to be plotted
            (for debugging)
    :return: a, a (2,) np.array of the shift
    """

    # Data coercsion, make sure we know what we're working wtih
    first = dataList[0]
    if isinstance(first, Absorbance):
        first = first.proc_data
    second = dataList[1]
    if isinstance(second, Absorbance):
        second = second.proc_data
    if plot:
        # Keep a reference to whatever plot is open at call-time
        # Useful if the calling script has plots before and after, as
        # omitting this will cause future plots to be added to figures here
        firstFig = plt.gcf()
        plt.figure("Stitcher")
        # Plot the raw input data
        plt.plot(*first.T)
        plt.plot(*second.T)

    # Algorithm is set up such that the "second" data set spans the
    # higher domain than first. Need to enforce this, and remember it
    # so the correct shift is applied
    flipped = False
    if max(first[:, 0]) > max(second[:, 0]):
        flipped = True
        first, second = second, first

    def fitter(p, shiftable, immutable):
        # designed to over

        # Get the shifts
        dx = p[0]
        dy = p[1]

        # Don't want pass-by-reference nonsense, recast our own refs
        shiftable = np.array(shiftable)
        immutable = np.array(immutable)

        # Shift the data set
        shiftable[:, 1] += dy
        shiftable[:, 0] += dx

        # Create an interpolator. We want a
        # direct comparision for subtracting the two functions
        # Different spec grating positions have different wavelengths
        # so they're not directly comparable.
        shiftF = spi.interp1d(*shiftable.T)

        # Find the bounds of where the two data sets overlap
        overlap = (min(shiftable[:, 0]), max(immutable[:, 0]))
        print("overlap", overlap)

        # Determine the indices of the immutable function
        # where it overlaps. argwhere returns 2-d thing,
        # requiring the [0] at the end of each call
        fOlIdx = (min(np.argwhere(immutable[:, 0] >= overlap[0]))[0],
                  max(np.argwhere(immutable[:, 0] <= overlap[1]))[0])
        print("fOlIdx", fOlIdx)

        # Get the interpolated values of the shiftable function at the same
        # x-coordinates as the immutable case
        newShift = shiftF(immutable[fOlIdx[0]:fOlIdx[1], 0])

        if plot:
            plt.plot(*immutable[fOlIdx[0]:fOlIdx[1], :].T, marker='o', label="imm", markersize=10)
            plt.plot(immutable[fOlIdx[0]:fOlIdx[1], 0], newShift, marker='o', label="shift")
        imm = immutable[fOlIdx[0]:fOlIdx[1], 1]
        shift = newShift
        return imm - shift

    a, _, _, msg, err = spo.leastsq(fitter, [0.0001, 0.01 * max(first[:, 1])], args=(second, first), full_output=1)
    # print "a", a
    if plot:
        # Revert back to the original figure, as per top comments
        plt.figure(firstFig.number)

    # Need to invert the shift if we flipped which
    # model we're supposed to move
    if flipped: a *= -1

    return a


def integrateData(data, t1, t2, ave=False):
    """
    Integrate a discrete data set for a
    given time period. Sums the data between
    the given bounds and divides by dt. Optional
    argument to divide by T = t2-t1 for calculating
    averages.

    data = 2D array. data[:,0] = t, data[:,1] = y
    t1 = start of integration
    t2 = end of integration


    if data is a NxM, with M>=3, it will take the
    third column to be the errors of the points,
    and return the error as the quadrature sum
    """
    t = data[:, 0]
    y = data[:, 1]
    if data.shape[0] >= 3:
        errors = data[:, 2]
    else:
        errors = np.ones_like(y) * np.nan

    gt = set(np.where(t > t1)[0])
    lt = set(np.where(t < t2)[0])

    # find the intersection of the sets
    vals = list(gt & lt)

    # Calculate the average
    tot = np.sum(y[vals])
    error = np.sqrt(np.sum(errors[vals] ** 2))

    # Multiply by sampling
    tot *= (t[1] - t[0])
    error *= (t[1] - t[0])

    if ave:
        # Normalize by total width if you want an average
        tot /= (t2 - t1)
        errors /= (t2 - t1)
    if not np.isnan(error):
        return tot, error
    return tot


def fourier_prep(x_vals, y_vals, num=None):
    """
    This function will take a Nx2 array with unevenly spaced x-values and make
    them evenly spaced for use in fft-related things.

    And remove nans!
    """
    y_vals = handle_nans(y_vals)
    spline = spi.interp1d(x_vals, y_vals,
                          kind='linear')  # for some reason kind='quadratic' doesn't work? returns all nans
    if num is None:
        num = len(x_vals)
    even_x = np.linspace(x_vals[0], x_vals[-1], num=num)
    even_y = spline(even_x)
    # even_y = handle_nans(even_y)
    return even_x, even_y


def handle_nans(y_vals):
    """
    This function removes nans and replaces them with linearly interpolated
    values.  It requires that the array maps from equally spaced x-values.
    Taken from Stack Overflow: "Interpolate NaN values in a numpy array"
    """
    nan_idx = np.isnan(y_vals)
    my_lambda = lambda x: x.nonzero()[0]  # Returns the indices where Trues reside
    y_vals[nan_idx] = np.interp(my_lambda(nan_idx), my_lambda(~nan_idx), y_vals[~nan_idx])
    return y_vals


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
                       # This is 1 because the peak picker function was calling the 10th order the 9th
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

def get_data_and_header(fname, returnOrigin = False):
    """
    Given a file to a raw data file, returns the data
    and the json decoded header.

    Can choose to return the origin header as well
    :param fname: Filename to open
    :return: data, header (dict)
    """
    with open(fname) as fh:
        line = fh.readline()
        header_string = ''
        while line[0]=='#':
            header_string += line[1:]
            line = fh.readline()

        # image files don't have an origin header
        if not "Images" in fname:
            oh = line
            # last readline in loop removes first line in Origin Header
            # strip the remaining two
            oh += fh.readline()
            oh += fh.readline()[:-1] #remove final \n

        # data = np.genfromtxt(fh, delimiter=',')
    data = np.genfromtxt(fname, delimiter=',')

    header = json.loads(header_string)

    if returnOrigin:
        return data, header, oh
    return data, header

def natural_glob(*args):
    # glob/python sort alphabetically, so 1, 10, 11, .., 2, 21,
    # but I sometimes wnat "natural" sorting: 1, 2, 3, ..., 10, 11, 12, ..., 20, 21, 21 ...
    # There's tons of stack overflows, so I grabbed one of them. I put it in here
    # because I use it all the damned time. I also almost always use it when
    # glob.glob'ing, so just internally do it that way
    #
    # This is taken from
    # https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    import re
    def atoi(text):
        try:
            return int(text)
        except ValueError:
            return text
        # return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split('(-?\d+)', text)]

    return sorted(glob.glob(os.path.join(*args)), key=natural_keys)

def convertTime(timeStr):
    """
    The data file headers have the timestamp of data collection. Sometimes you want to
    convert that to numbers for data's sake, but I constantly forget the functions
    to convert it from the time-stamp string. So here you go
    :param timeStr: the time as a string from the data file
    :return: int of the time since the epoch
    """
    import time
    return time.mktime(time.strptime(timeStr, "%x %X%p"))


# photonConverter[A][B](x):
#    convert x from A to B.
photon_converter = {
    "nm":         {"nm": lambda x: x,           "eV": lambda x:1239.84/x,            "wavenumber": lambda x: 10000000./x},
    "eV":         {"nm": lambda x: 1239.84/x,   "eV": lambda x: x,                   "wavenumber":lambda x: 8065.56 * x},
    "wavenumber": {"nm": lambda x: 10000000./x, "eV": lambda x: x/8065.56, "wavenumber": lambda x: x}
}

####################
# Smoothing functions
####################

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


####################
# Complete functions
####################

def proc_n_plotPMT(folder_path, plot=False, confirm_fits=False, save=None, verbose=False, **kwargs):
    """
    This function will take a pmt object, process it completely.

    :rtype: list of HighSidebandPMT
    """
    pmt_data = pmt_sorter(folder_path, plot_individual=plot)

    index = 0
    for spectrum in pmt_data:
        spectrum.integrate_sidebands(verbose=verbose, **kwargs)
        spectrum.laser_line(verbose=verbose, **kwargs)  # This function is broken
        # because process sidebands can't handle the laser line
        # print spectrum.full_dict
        if plot:
            plt.figure('PMT data')
            for sb, elem in list(spectrum.sb_dict.items()):
                plt.errorbar(elem[:, 0], elem[:, 1], elem[:, 2],
                             marker='o', label="{} {}".format(spectrum.parameters["series"],sb))
            plt.figure('Sideband strengths')
            plt.yscale("log")
            plt.errorbar(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3], spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
        if plot and confirm_fits:
            plt.figure('PMT confirm fits')
            for elem in list(spectrum.sb_dict.values()):
                plt.errorbar(elem[:, 0], elem[:, 1], elem[:, 2], marker='o')
            plt.errorbar(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3], spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
            plt.ylim([-0.005, 0.025])
        if type(save) is tuple:
            spectrum.save_processing(save[0], save[1], index=index)
            index += 1
        elif isinstance(save, str):
            dirr = os.path.dirname(save) if os.path.dirname(save) else '.' # if you just pass a filename tos ave
            spectrum.save_processing(os.path.basename(save), dirr,
                                     index=index)
            index += 1
    if plot:
        plt.legend()
    return pmt_data


def proc_n_plotCCD(folder_path, offset=None, plot=False, confirm_fits=False,
                   save=None, keep_empties = False, verbose=False, **kwargs):
    """
    This function will take a list of ccd files and process it completely.
    save_name is a tuple (file_base, folder_path)

    keep_empties: If True, keep the HighSidebandCCD object in the list if no sidebands
    are found. Else, cut it off.

    The cutoff of 8 is too high, but I don't know what to change it to
    :rtype: list of HighSidebandCCD
    """
    if isinstance(folder_path, list):
        file_list = folder_path
    else:
        # if verbose:
            # print "Looking in:", os.path.join(folder_path, '*seq_spectrum.txt')
        # file_list = glob.glob(os.path.join(folder_path, '*seq_spectrum.txt'))
        file_list = natural_glob(folder_path, '*seq_spectrum.txt')
        # if verbose:
            # print "found these files:", "\n".join([os.path.basename(ii) for ii in file_list])
    raw_list = []
    for fname in file_list:
        raw_list.append(HighSidebandCCD(fname, spectrometer_offset=offset))

    index = 0
    for spectrum in raw_list:
        try:
            spectrum.guess_sidebands(verbose=verbose, plot=plot)
        except RuntimeError:
            print("\n\n\nNo sidebands??\n\n")
            # No sidebands, say it's empty
            if not keep_empties:
                raw_list.pop(raw_list.index(spectrum))
            continue
        try:
            spectrum.fit_sidebands(plot=plot, verbose=verbose)
        except RuntimeError:
            print("\n\n\nNo sidebands??\n\n")
            # No sidebands, say it's empty
            if not keep_empties:
                raw_list.pop(raw_list.index(spectrum))
            continue
        if "calculated NIR freq (cm-1)" not in list(spectrum.parameters.keys()):
            spectrum.infer_frequencies()
        if plot:
            plt.figure('CCD data')
            plt.errorbar(spectrum.proc_data[:, 0], spectrum.proc_data[:, 1], spectrum.proc_data[:, 2],
                         label=spectrum.parameters['series'])
            plt.legend()
            # plt.yscale('log')
            plt.figure('Sideband strengths')
            plt.errorbar(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3], spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
            plt.legend()
            plt.yscale('log')
        if plot and confirm_fits:
            plt.figure('CCD confirm fits')
            plt.plot(spectrum.proc_data[:, 0], spectrum.proc_data[:, 1],# spectrum.proc_data[:, 2],
                         label=spectrum.parameters['series'])
            plt.plot(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3] / spectrum.sb_results[:, 5],# spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
            plt.legend()
            plt.ylim([-0.1, 1])
        if type(save) is tuple:
            spectrum.save_processing(save[0], save[1],
                                     marker=spectrum.parameters["series"].replace(
                                         r"/", "p"),
                                     index=index)
            index += 1
        elif isinstance(save, str):
            # print "DEBUG: trying to save CCD with ", os.path.dirname(save),'_at_', os.path.basename(save)
            spectrum.save_processing(os.path.basename(save), os.path.dirname(save),
                                     marker=spectrum.parameters["series"].replace(
                                         r"/", "p"),
                                     index=index)
            index += 1
    return raw_list


def create_full_spectra(folder_path, skipLaser = True, *args, **kwargs):
    """
    Given the folder path of raw data (where the PMT data is held in the subfolder "PMT"),
    scale all the data to create a raw comb spectra.
    :param folder_path:
    :param args:
    :param kwargs:
    :return:
    """
    output = np.empty((0,2))
    # have proc_n_plot do all the integrating for the sbs
    pmt = proc_n_plotPMT(os.path.join(folder_path, "PMT"))

    ccd_file_list = glob.glob(os.path.join(folder_path, '*seq_spectrum.txt'))
    ccd_list = [HighSidebandCCD(fname) for fname in ccd_file_list]




    for pmtsb in sorted(pmt[0].sb_dict.keys()):
        if skipLaser and pmtsb == 0: continue
        data = pmt[0].sb_dict[pmtsb]
        try:
            print(pmtsb, pmt[0].full_dict[pmtsb])
        except:
            continue
        output = np.row_stack((output, np.abs(data[:,[0,1]])))
        output = np.row_stack((output, [np.nan, np.nan]))

    # insert the pmt so I can iterate over scaling consecutive pairs
    ccd_list.insert(0, pmt[0])

    # make sure all things get scaled down by the factors before them
    runningRatio = 1
    for idx, ccd in enumerate(ccd_list[1:]):
        ccd.guess_sidebands()
        ccd.fit_sidebands()
        ratio = [1, 1]

        stitch_hsg_dicts(ccd_list[idx], ccd, need_ratio = True, ratios=ratio)

        print("new ratio", ratio)
        runningRatio *= ratio[1]
        ccd.proc_data[:,1]*=runningRatio

        output = np.row_stack((output, np.abs(ccd.proc_data[:,[0,1]])))
        output = np.row_stack((output, [np.nan, np.nan]))

    offsetEnergy = (output[:,0] - pmt[0].full_dict[0][0])*1e3
    print(offsetEnergy.shape, output.shape)
    output = np.column_stack((output[:,0], offsetEnergy.T, output[:,1]))

    return output
