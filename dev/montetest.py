import monteCarloObject as mco
import numpy as np
import Stele
import Stele.QWPProcessing as qwp


def get_alphagamma(
        file, observedSidebands, crystalAngle,
        saveFileName, save_results=False):
    """
    This function will take a folder of polarimetry scans and produce
    alphas and gammas with error.

    :param file: String of folder name containing 4 polarimetry scans
    :param observedSidebands: np array of observed sidebands. Data will be
        cropped such that these sidebands are included in everything.
    :param crystalAngle: (Float) Angle of the sample from the 010 crystal face
    :saveFileName: Str of what you want to call the text files to be saved
    :save_results: Boolean controls if things are saved to txt files.
        Currently saves DOP, alpha, gamma, J matrix, T matrix, Fan, and Matrix
        Relations

    :return: alphaFull, gammaFull
    These are matrices of the form
        -1  |   NIRa   |   dNIR err  |  NIRa2   |  dNIRa err  |    ...
     sb1    | SB Data  | SB Data err | SB Data  | SB Data err |    ...
     sb2    |   ...    |     ...     |
     .
     .

    """

    # Make a list of folders for the fan data
    datasets = Stele.natural_glob(file, "*")

    # FanCompiler class keeps alpha and gamma data together for different pols
    fanData = qwp.expFanCompiler.FanCompiler(
            observedSidebands, keepErrors=True)

    for data in datasets:
        laserParams, rawData = Stele.hsg_combine_qwp_sweep(data)
        _, fitDict = Stele.proc_n_fit_qwp_data(
            rawData, vertAnaDir=False,
            laserParams=laserParams, wantedSbs=observedSidebands)

        # Add the new alpha and gamma sets to the fan class
        fanData.addSet(fitDict)

    # Building the fan compiler causes it to make  2D np arrays with the alpha
    #   and gamma angles where each column is a different alpha_NIR excitation
    alphaData, gammaData, _ = fanData.build(withErrors=True)

    # save the alpha and gamma results
    if save_results:
        fanData.buildAndSave(saveFileName + "_{}.txt")

    return alphaData, gammaData

# These are some things to set before hand. I picked this from some data we
#   took a few weeks ago.


observedSidebands = np.arange(8, 32, 2)
crystalAngle = 16

saveFileName = "12-05_alphasgammas"
# Feel free to change this one, don't change the other two

alphas, gammas = get_alphagamma(
    r"Fan Data", observedSidebands, 16, saveFileName, save_results=False)

monte = mco.monteCarlo(alphaData=alphas,
                       gammaData=gammas,
                       folder_name='theta9001',
                       observedSidebands=(8, 30))
