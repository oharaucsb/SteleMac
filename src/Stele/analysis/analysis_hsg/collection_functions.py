import os
import io
import glob
import copy
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import itertools as itt
import analysis_hsg.complete_functions as hsg_complete
import analysis_hsg.helper_functions as hsg_help

np.set_printoptions(linewidth=500)


def hsg_combine_spectra(spectra_list, verbose=False, **kwargs):
    """
    Smooshes an HSG spectrum together using stitch_hsg_dicts.

    This function is all about smooshing different parts of the same hsg
    spectrum together.  It takes a list of HighSidebandCCD spectra and turns
    the zeroth spec_step into a FullHighSideband object.  It then uses the
    function stitch_hsg_dicts over and over again for the smooshing.

    Input:
    spectra_list = list of HighSidebandCCD objects that have sideband spectra
                   larger than the spectrometer can see.

    Returns:
    -------
    good_list = A list of FullHighSideband objects that have been combined as
                much as can be.

    :param spectra_list: randomly-ordered list of HSG spectra, some of which
        can be stitched together
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
        current_steps = spec_steps.get(elem.parameters["series"], [])
        current_steps.append(elem.parameters["spec_step"])
        spec_steps[elem.parameters["series"]] = current_steps
    if verbose:
        print("I found these spec steps for each series:")
        print("\n\t".join("{}: {}".format(*ii) for ii in spec_steps.items()))

    # sort the list of spec steps
    for series in spec_steps:
        spec_steps[series].sort()

    same_freq = lambda x, y: x.parameters["fel_lambda"] == y.parameters["fel_lambda"]

# TODO: correct to proper loop structure
#       it appears to remove items from the list and put them into temp until
#       it runs out of elements to pop, raising an exception and exiting
#       exception usage in this case appears to be correctly used, loop should
#       terminate at end of range for spectra_list
#       HOWEVER, as for what this does, it appears to take the last element in
#       spectra_list and assign it to temp, then delete spectra list?
    for index in range(len(spectra_list)):
        try:
            temp = spectra_list.pop(0)
            if verbose:
                print("\nStarting with this guy", temp, "\n")
        except Exception:
            break

        good_list.append(FullHighSideband(temp))

        counter = 1
        temp_list = list(spectra_list)
        for piece in temp_list:
            if verbose:
                print(
                    "\tchecking this spec_step",
                    piece.parameters["spec_step"], end=' '
                    )
                print(", the counter is", counter)
            if not same_freq(piece, temp):
                if verbose:
                    print("\t\tnot the same fel frequencies ({} vs {})".format(
                        piece.parameters["fel_lambda"],
                        temp.parameters["fel_lambda"]))
                continue
            if temp.parameters["series"] == piece.parameters["series"]:
                if piece.parameters["spec_step"] \
                        == spec_steps[temp.parameters["series"]][counter]:
                    if verbose:
                        print("I found this one", piece)
                    counter += 1
                    good_list[-1].add_CCD(piece, verbose=verbose, **kwargs)
                    spectra_list.remove(piece)
                else:
                    print(
                        "\t\tNot the right spec step?",
                        type(piece.parameters["spec_step"])
                        )

            else:
                if verbose:
                    print("\t\tNot the same series ({} vs {}".format(
                        piece.parameters["series"], temp.parameters["series"]))
        good_list[-1].make_results_array()
    return good_list


def hsg_combine_spectra_arb_param(
        spectra_list, param_name="series", verbose=False):
    """
    Smooshes like combine_spectra, but allows you to pick what is the "same".

    This function is all about smooshing different parts of the same hsg
    spectrum together.  It takes a list of HighSidebandCCD spectra and turns
    the zeroth spec_step into a FullHighSideband object.  It then uses the
    function stitch_hsg_dicts over and over again for the smooshing.

    This is different than hsg_combine_spectra in that you pass which criteria
    distinguishes the files to be the "same". Since it can be any arbitrary
    value, things won't be exactly the same (field strength will never be
    identical between images). It will start with the first (lowest) spec step,
    then compare the number of images in the next step. Whichever has

    Input:
    spectra_list = list of HighSidebandCCD objects that have sideband spectra
                   larger than the spectrometer can see.

    Returns:
    -------
    good_list = A list of FullHighSideband objects that have been combined as
                much as can be.

    :param spectra_list: randomly-ordered list of HSG spectra, some of which
        can be stitched together
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
        if elem in already_added:
            continue
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
                good_list[-1].parameters["spec_step"], good_list[-1].parameters["series"]
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
            best_values = np.array(
                [x.parameters[param_name]["mean"] for x in best_match])
            best_std = np.array(
                [x.parameters[param_name]["std"] for x in best_match])
            new_value = np.average(best_values, weights=best_std)
            new_std = np.sqrt(
                np.average((best_values-new_value)**2, weights=best_std))

        good_list[-1].parameters[param_name] = {
            "mean": new_value,
            "std": new_std
        }
    return good_list


def pmt_sorter(folder_path, plot_individual=True):
    """
    Turn a folder of PMT files into a list of HighSidebandPMT objects.

    This function will be fed a folder with a bunch of PMT data files in it.
    The folder should contain a bunch of spectra with at least one sideband in
    them, each differing by the series entry in the parameters dictionary.

    This function will return a list of HighSidebandPMT objects.

    :param folder_path: Path to a folder containing a bunch of PMT data, can be
                        part of a parameter sweep
    :type folder_path: str
    :param plot_individual: Whether to plot each sideband itself
    :return: A list of all the possible hsg pmt spectra, organized by series
        tag
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
            plt.errorbar(
                elem[:, 0], elem[:, 1], elem[:, 2],
                marker='o',
                label="{} {}, {}.{} ".format(
                    spec.parameters["series"], spec.initial_sb,
                    spec.parameters["pm_hv"],
                    't' if spec.parameters.get(
                        "photon counted", False) else 'f')
                )

    for sb_file in file_list:
        temp = HighSidebandPMT(sb_file)
        plot_sb(temp)
        try:
            for pmt_spectrum in pmt_list:  # pmt_spectrum is a pmt object
                if temp.parameters['series'] \
                        == pmt_spectrum.parameters['series']:
                    pmt_spectrum.add_sideband(temp)
                    break
            else:  # this will execute IF the break was NOT called
                pmt_list.append(temp)
        except Exception:
            pmt_list.append(temp)
    for pmt_spectrum in pmt_list:
        pmt_spectrum.process_sidebands()
    return pmt_list


def stitch_abs_results(main, new):
    raise NotImplementedError


def hsg_combine_qwp_sweep(
        path, loadNorm=True, save=False, verbose=False, skipOdds=True):
    """
    Process polarimetry data into a matrix of sb strength, QWP angle, sb #.

    Given a path to data taken from rotating the QWP (doing polarimetry),
    process the data (fit peaks), and parse it into a matrix of sb strength vs
    QWP angle vs sb number.

    By default, saves the file into "Processed QWP Dependence"

    Return should be passed directly into fitting

        -1  |      SB1    |  SB1 |    SB2    |  SB2 | ... | ... | SBn | SBn |
     angle1 | SB Strength |SB err|SB Strength|SB Err|
     angle2 |     ...     |   .  |
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
        Load incident NIR data and collect header information.

        Helper function for loading the data and getting the header information
            for incident NIR stuff
        :param fname:
        :return:
        """
        if isinstance(fname, str):
            if loadNorm:
                ending = "_norm.txt"
            else:
                ending = "_snip.txt"
            header = ''
            with open(os.path.join(
                    "Processed QWP Dependence", fname + ending)) as fh:
                ln = fh.readline()
                while ln[0] == '#':
                    header += ln[1:]
                    ln = fh.readline()
            data = np.genfromtxt(
                os.path.join("Processed QWP Dependence", fname + ending),
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
        return data, float(header["lAlpha"]), float(header["lGamma"]), \
            float(header["nir"]), float(header["thz"])

    try:
        sbData, lAlpha, lGamma, nir, thz = getData(path)
        # TODO: correct control structure to not be reliant on Exception
    except Exception:
        # Do the processing on all the files
        specs = hsg_complete.proc_n_plotCCD(
            path, keep_empties=True, verbose=verbose)

        for sp in specs:
            try:
                sp.parameters["series"] = round(float(
                    sp.parameters["rotatorAngle"]), 2)
            except KeyError:
                # Old style of formatting
                sp.parameters["series"] = round(float(
                    sp.parameters["detectorHWP"]), 2)
        specs = hsg_combine_spectra(specs, ignore_weaker_lowers=False)
        if not save:
            # If you don't want to save them, set everything up for doing Bytes
            # objects to replacing saving files
            full, snip, norm = io.BytesIO(), io.BytesIO(), io.BytesIO()
            if "nir_pola" not in specs[0].parameters:
                # in the olden days. Force them. Hopefully making them outside
                # of ±360 makes it obvious
                specs[0].parameters["nir_pola"] = 361
                specs[0].parameters["nir_polg"] = 361
            keyName = "rotatorAngle"
            if keyName not in specs[0].parameters:
                # from back before I changed the name
                keyName = "detectorHWP"

            hsg_help.save_parameter_sweep(
                specs, [full, snip, norm], None, keyName, "deg",
                wanted_indices=[3, 4],
                header_dict={
                    "lAlpha": specs[0].parameters["nir_pola"],
                    "lGamma": specs[0].parameters["nir_polg"],
                    "nir": specs[0].parameters["nir_lambda"],
                    "thz": specs[0].parameters["fel_lambda"],
                    },
                only_even=skipOdds
                )

            if loadNorm:
                sbData, lAlpha, lGamma, nir, thz = getData(norm)
            else:
                sbData, lAlpha, lGamma, nir, thz = getData(snip)
        else:
            hsg_help.save_parameter_sweep(
                specs, os.path.basename(path), "Processed QWP Dependence",
                "rotatorAngle", "deg", wanted_indices=[3, 4],
                header_dict={
                    "lAlpha": specs[0].parameters["nir_pola"],
                    "lGamma": specs[0].parameters["nir_polg"],
                    "nir": specs[0].parameters["nir_lambda"],
                    "thz": specs[0].parameters["fel_lambda"],
                    },
                only_even=skipOdds
                )
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
    foundSidebands = np.array(
        sorted([float(ii) for ii in set(sbData[2]) if ii]))

    # Remove first 3 rows, which are strings for origin header, & cast to float
    sbData = sbData[3:].astype(float)

    # double the sb numbers (to account for sb strength/error) and add a dummy
    # number so the array is the same shape
    foundSidebands = np.insert(
        foundSidebands, range(len(foundSidebands)), foundSidebands)
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


def proc_n_fit_qwp_data(
        data, laserParams=dict(), wantedSBs=None, vertAnaDir=True, plot=False,
        save=False, plotRaw=lambda sbidx, sbnum: False, series='', eta=None,
        **kwargs
        ):
    """
    Fit a set of sideband data vs QWP angle to get the stoke's parameters.

    :param data: data in the form of the return of hsg_combine_qwp_sweep
    :param laserParams: dictionary of the parameters of the laser, the angles
        and frequencies. See function for expected keys. I don't think the
        errors are used (except for plotting?), or the wavelengths (but left in
        for potential future use (wavelength dependent stuff?))
    :param wantedSBs: List of the wanted sidebands to fit out.
    :param vertAnaDir: direction of the analzyer. True if vertical, false if
        horizontal.
    :param plot: True/False to plot alpha/gamma/dop. Alternatively, a list of
        "a", "g", "d" to only plot selected ones
    :param save: filename to save the files. Accepts BytesIO
    :param plotRaw: callable that takes an index of the sb and sb number,
        returns true to plot the raw curve
    :param series: a string to be put in the header for the origin files
    :param eta: a function to call to calculate the desired retardance. Input
        will be the SB order.

    if saveStokes is in kwargs and False, it will not save the stokes
        parameters, since I rarely actually use them.
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
    lAlpha, ldAlpha, lGamma, ldGamma, lDOP, ldDOP = \
        defaultLaserParams["lAlpha"], \
        defaultLaserParams["ldAlpha"], \
        defaultLaserParams["lGamma"], \
        defaultLaserParams["ldGamma"], \
        defaultLaserParams["lDOP"], \
        defaultLaserParams["ldDOP"]
    allSbData = data
    angles = allSbData[1:, 0]

    allSbData = allSbData[:, 1:]  # trim out the angles

    if wantedSBs is None:
        # set to get rid of duplicates, 1: to get rid of the -1 used for
        # getting arrays the right shape
        wantedSBs = set(allSbData[0, 1:])

    if eta is None:
        """
        It might be easier for the end user to do this by passing
        eta(wavelength) instead of eta(sborder), but then this function would
        need to carry around wavelengths, which is extra work. It could convert
        between NIR/THz wavelengths to SB order, but it's currently unclear
        whether you'd rather use what the WS6 claims, or what the sidebands
        say, and you'd probably want to take the extra step to ensure the SB
        fit rseults if using the spectromter wavelengths. In general, if you
        have a function as etal(wavelength), you'd probably want to pass this
        as eta = lambda x: etal(1239.84/(nirEv + x*THzEv))
        assuming nirEv/THzEv are the photon energies of the NIR/THz.
        """
        eta = lambda x: 0.25

    # allow pasing a flag it ignore odds. I think I generally do, so set it to
    # default to True
    skipOdds = kwargs.get("skip_odds", True)

    # Make an array to keep all of the sideband information.
    # Start it off by keeping the NIR information
    #   (makes for easier plotting into origin)
    sbFits = [[0] + [-1] * 8 + [lAlpha, ldAlpha, lGamma, ldGamma, lDOP, ldDOP]]
    # Also, for convenience, keep a dictionary of the information.
    # This is when I feel like someone should look at porting this
    # over to pandas
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
        if sbNum not in wantedSBs:
            continue
        if skipOdds and sbNum % 2:
            continue
        sbData = allSbData[1:, sbIdx]
        sbDataErr = allSbData[1:, sbIdx + 1]
        p0 = [1, 1, 0, 0]

        etan = eta(sbNum)
        try:
            p, pcov = curve_fit(
                makeCurve(etan, vertAnaDir), angles, sbData, p0=p0
                )
        except ValueError:
            # This is getting tossed around, especially when looking at noisy
            # data, especially with the laser line, and it's fitting erroneous
            # values.
            # Ideally, I should be cutting this out and not even returning
            # them, but that's immedaitely causing
            p = np.nan*np.array(p0)
            pcov = np.eye(len(p))

        if plot and plotRaw(sbIdx, sbNum):
            plt.figure("All Curves")
            plt.errorbar(
                angles, sbData, sbDataErr, 'o-', name=f"{series}, {sbNum}")
            fineAngles = np.linspace(angles.min(), angles.max(), 300)
            plt.plot(fineAngles, makeCurve(etan, vertAnaDir)(fineAngles, *p))
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
        variance = \
            (d2 ** 2 * S1 ** 2 + d1 ** 2 * S2 ** 2) \
            / (S1 ** 2 + S2 ** 2) ** 2
        thisData.append(np.sqrt(variance) * 180. / np.pi)

        sbFitsDict["alpha"].append([sbNum, thisData[-2], thisData[-1]])

        # append gamma value
        thisData.append(
            np.arctan2(S3, np.sqrt(S1 ** 2 + S2 ** 2)) / 2 * 180. / np.pi)
        # append gamma error
        variance = (
            d3 ** 2 * (S1 ** 2 + S2 ** 2) ** 2
            + (d1 ** 2 * S1 ** 2 + d2 ** 2 * S2 ** 2) * S3 ** 2
            ) \
            / ((S1 ** 2 + S2 ** 2) * (S1 ** 2 + S2 ** 2 + S3 ** 2) ** 2)
        thisData.append(np.sqrt(variance) * 180. / np.pi)
        sbFitsDict["gamma"].append([sbNum, thisData[-2], thisData[-1]])

        # append degree of polarization
        thisData.append(np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / S0)
        variance = (
            (
                d1 ** 2 * S0 ** 2 * S1 ** 2
                + d0 ** 2 * (S1 ** 2 + S2 ** 2 + S3 ** 2) ** 2
                + S0 ** 2 * (d2 ** 2 * S2 ** 2 + d3 ** 2 * S3 ** 2)
            )
            / (S0 ** 4 * (S1 ** 2 + S2 ** 2 + S3 ** 2))
        )
        thisData.append(np.sqrt(variance))
        sbFitsDict["DOP"].append([sbNum, thisData[-2], thisData[-1]])

        sbFits.append(thisData)

    sbFits = np.array(sbFits)
    sbFitsDict = {k: np.array(v) for k, v in sbFitsDict.items()}

    # to fit all other files for easy origin importing
    origin_header = "#\n"*100
    origin_header += 'Sideband,S0,S0 err,S1,S1 err,S2,S2 err,S3,S3 err,' \
        + 'alpha,alpha err,gamma,gamma err,DOP,DOP err\n'
    origin_header += 'Order,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,arb.u,' \
        + 'arb.u,deg,deg,deg,deg,arb.u.,arb.u.\n'
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
        np.savetxt(
            save, np.array(sbFitsSave), delimiter=',', header=origin_header,
            comments='', fmt='%.6e'
            )
    if plot:
        plt.figure("alpha")
        plt.errorbar(sbFitsDict["alpha"][:, 0],
                     sbFitsDict["alpha"][:, 1],
                     sbFitsDict["alpha"][:, 2],
                     'o-', name=series
                     )
        plt.figure("gamma")
        plt.errorbar(sbFitsDict["gamma"][:, 0],
                     sbFitsDict["gamma"][:, 1],
                     sbFitsDict["gamma"][:, 2],
                     'o-', name=series
                     )
    return sbFits, sbFitsDict
