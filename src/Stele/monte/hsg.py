"""
Functions to find for this file
hsg.natural_glob
hsg.proc_n_fit_qwp_data
hsg.hsg_combine_qwp_sweep
"""
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
