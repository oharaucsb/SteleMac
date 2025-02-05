import numpy as np
import os
import Stele as hsg


class FanCompiler(object):
    """
    Helper class for compiling the data of a polarimetry NIR alpha sweep.
    This class helps to normalize the datasets: it will make sure each slice of
    a different NIR_alpha has the same sideband orders, preventing issues when
    not all data sets saw the same number of orders.

    Typical use scenario:

    datasets = [ ... list of folders, each being a dataset of different NIR
    alphas ...] outputs = FanComilier(<whatever sideband orders you want
    compiled>)
    for data in datasets:
        laserParams, rawData = hsg.hsg_combine_qwp_sweep(folder, save=False,
        verbose=False) _, fitDict = hsg.proc_n_fit_qwp_data(rawData,
        laserParams, vertAnaDir="VAna" in folder, series=folder)
        outputs.addSet(nira, fitDict)

    outputs.buildAndSave(fname)

    This is put together in the static method, fromDataFolder, where you pass
    the folder which contains the folders of polarimetry data

    build() will return a dictionary where each key is a diferent angle
    parameter The values of the dict are the 2D datasets from polarimetry

    """
    def __init__(self, wantedSBs, keepErrors=False, negateNIR=True):
        """

        :param wantedSBs:
        :param keepErrors: Whether the errors in the angles/values are kept in
            the data sets or not. Defaults to false because they are often not
            used (this class getting passed to FanDiagrams and such)
        :param negateNIR: flag for whether to negate the NIR alpha value.
            Currently, this is done because the PAX views -z direction,\
            while home-built views +z (
        with NIR)
        """
        self.want = np.array(wantedSBs)

        # I guess you could make this a kwarg to the class and pass it, but
        # I don't think it really matters here
        keys = ["S0", "S1", "S2", "S3", "alpha", "gamma", "DOP"]

        # Keep an array for each of the datasets directly
        self.outputArrays = {ii: wantedSBs.reshape(-1, 1) for ii in keys}

        # Tracking the NIR alpha/gamma's which are provided to the compiler
        self.nirAlphas = []
        self.nirGammas = []
        # A flag used when loading data to determine whether or not to keep the
        # errors along
        self._e = keepErrors

        # Flag for whether the NIR alphas should be negated or not. It prints
        # an error as it seemed like a bad thing to need to arbitrarily negate
        # an angle, before it was realized the PAX and polarimeter measure
        # different reference frames
        self._n = 1
        if negateNIR:
            print("WARNING: NEGATING NIR ALPHA AND GAMMA")
            self._n = -1

    @staticmethod
    def fromDataFolder(folder, wantedSBs, keepErrors=False, negateNIR=True,
                       eta=None, doNorms=False):
        """
        Create a fan compiler by passing the data path. Handles looping through
            the folder's sub-folders to find
        :param folder: The folder to search through. Alternatively, if it's a
        list/iterable, iterate through that instead. Useful if external code is
        directly removing sets of data.
        :return:
        """
        comp = FanCompiler(wantedSBs, keepErrors, negateNIR)
        # If it's a string, assume a single path that wants to be searached
        if isinstance(folder, str):
            wantFolders = hsg.natural_glob(folder, "*")
        else:
            # Otherwise, assume they've passed an iterable to search through
            wantFolders = folder

        for nirFolder in wantFolders:
            # Provide ways to skip over data sets by mis-naming/flagging them
            if "skip" in nirFolder.lower() or "bad" in nirFolder.lower():
                continue
            laserParams, rawData = hsg.hsg_combine_qwp_sweep(
                nirFolder, save=False, verbose=False, loadNorm=doNorms)

            _, fitDict = hsg.proc_n_fit_qwp_data(
                rawData, laserParams, vertAnaDir="VAna" in nirFolder,
                series=nirFolder, eta=eta)
            comp.addSet(fitDict)
        return comp

    def addSet(self, dataSet):
        """ Assume it's passed from  proc_n_fit_qwp_data, assuming it is a dict
        with the keys and shape returned by that function"""

        # Keeps track of all the new data passed. Keys are the angles/S
        # parameters, values are arrays for each of those which will contain
        # the value of the parameters for a given sideband order
        newData = {ii: [] for ii in self.outputArrays}

        # new data needs to parsed to extract the relevant parameters for all
        # the sidebands specified in this FanCompiler's constructor. Data is
        # kept as a list of lists in `newData`

        if self._e:
            # keep track of the errors in everything
            nirAlpha = [self._n*dataSet["alpha"][0][1], dataSet["alpha"][0][2]]
            nirGamma = [self._n*dataSet["gamma"][0][1], dataSet["gamma"][0][2]]
            for sb in self.want:
                # the list(*[]) bullshit is to make sure a list gets appended,
                # not a numpy array. Further complicated because if the list
                # comprehension returns nothing, it doesn't append anything,
                # hence casting to a list to enforce an empty list gets
                # appended.
                [jj.append(list(*[ii[1:] for ii in dataSet[kk] if
                           ii[0] == sb]))
                 for kk, jj in newData.items()]
                if not newData["alpha"][-1]:
                    # no data was found, so the last element is an empty array.
                    # Force it to have elements with the same dimensions so it
                    # won't break numpy analysis below
                    for jj in newData.values():
                        jj[-1] = [np.nan, np.nan]
        else:
            # Even though it's a single number each, put them in lists so they
            # can be list.extended() and consistent with the error usage above
            nirAlpha = [self._n*dataSet["alpha"][0][1]]
            nirGamma = [self._n*dataSet["gamma"][0][1]]
            for sb in self.want:
                # Even though only one element is being kept (i.e. data without
                # error), it's still being placed inside a list to be
                # consistent with the use case with errors above
                [jj.append(list([ii[1] for ii in dataSet[kk] if ii[0] == sb]))
                 for kk, jj in newData.items()]
                # no data was found.
                if not newData["alpha"][-1]:
                    for jj in newData.values():
                        jj[-1] = [np.nan]

        for k, v in newData.items():
            self.outputArrays[k] = np.column_stack((self.outputArrays[k], v))

        # extending created lists accounts for keeping errors r not
        self.nirAlphas.extend(nirAlpha)
        self.nirGammas.extend(nirGamma)

    def build(self, withErrors=True):
        """
        Return only alpha, gamma, S0 parameters directly, for compatibility and
            ease.
        :param withErrors:
        :return:
        """
        data = self.buildAll()
        if not self._e:
            # You didn't keep track of errors when loading datasets, so just
            # return the data sets
            return data["alpha"], data["gamma"], data["S0"]

        if withErrors:
            return data["alpha"], data["gamma"], data["S0"]
        alphaData = np.column_stack((
            data["alpha"][:, 0], data["alpha"][:, 1::2]
            ))
        gammaData = np.column_stack((
            data["gamma"][:, 0], data["gamma"][:, 1::2]
            ))
        S0Data = np.column_stack((data["S0"][:, 0], data["S0"][:, 1::2]))
        return alphaData, gammaData, S0Data

    def buildAll(self):
        """
        a dict of d[<S0,S1,S2,S3,alpha, gamma, DOP>] where each item is a
            2D array:

                 -1  |   NIRa   |   dNIR err  |  NIRa2   |  dNIRa err  |    ...
              sb1    | SB Data  | SB Data err | SB Data  | SB Data err |    ...
              sb2    |   ...    |     ...     |
              .
              .

        where "SB Data" is the data corresponding to the key of the dict.

        Errors are not included if not passed in the fanCompiler constructor

        return["gamma"] replaces the first row with the NIR gamma values, which
        was useful when doing polarimetry with non-linearly polarized light
        (i.e. circular) Furthermore, these values get passed to the Jones
        matrix extraction stuff

        :return:
        """
        fullData = {ii: np.append([-1], self.nirAlphas) for ii in
                    self.outputArrays.keys()}

        for k, v in self.outputArrays.items():
            fullData[k] = np.row_stack((fullData[k], v))

        # insert the gamma values into that array for the NIR laser
        fullData["gamma"][0, 1:] = self.nirGammas
        return fullData

    def buildAndSave(self, fname, *args, saveStokes=False):
        """
        fname: filename to save to. Must have a least one string formatter
            position to allow for saving separate alpha/gamma/s0 files. *args
            are passed to any other formatting positions.
        :param fname:
        :param args:
        :param saveStokes: Pass true if you want to save the stokes files
        :return:
        """

        if os.path.dirname(fname) and not os.path.exists(
                os.path.dirname(fname)):

            os.mkdir(os.path.dirname(fname))

        oh = "#\n" * 100
        oh += "\n\n"

        fullDataA, fullDataG, fullDataS = self.build()

        outputs = self.buildAll()

        if saveStokes:
            saveEms = [ii for ii in outputs.keys()]
        else:
            saveEms = ["alpha", "gamma", "S0"]

        for saveEm in saveEms:
            np.savetxt(fname.format(saveEm, *args), outputs[saveEm], header=oh,
                       delimiter=',', comments='')
