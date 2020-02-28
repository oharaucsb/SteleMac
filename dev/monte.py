import Stele.ipg as pg
import numpy as np
import Stele as hsg
import Stele.QWPProcessing as qwp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import os

"""

Making this code that defines a function to take data from 4 polarimetry scans,
calculate alpha and gamma (and error), then find the Jone matrix.g

"""


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
    datasets = hsg.natural_glob(file, "*")

    # FanCompiler class keeps alpha and gamma data together for different pols
    fanData = qwp.expFanCompiler.FanCompiler(
            observedSidebands, keepErrors=True)

    for data in datasets:
        laserParams, rawData = hsg.hsg_combine_qwp_sweep(data)
        _, fitDict = hsg.proc_n_fit_qwp_data(
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

alphaData, gammaData = get_alphagamma(
    r"Fan Data", observedSidebands, 16, saveFileName, save_results=False)

# Now we need to calculate the Jones matrices.
# For this function you need to cut out errors, and just feed it an array of
# sb, alpha
# I leave that for you to figure out

# within alphadata & gammadata array, skip first line
# after that, each line is for a given side band,
# within that sideband, values are sideband|measure|error|measure|error|+2

# iterate first loop across the sidebands vertically
# iterate second loop for monte carlo within that band

# name of folder for all future results to be dumped to
saveFolder = 'theta9000'
# number of times to iterate the monte carlo
monteCarlo = 5000
# width of arrays of alphas and gammas
AGwidth = 4
# matrix for monte carlo results beginning with 2D slice of zeros for dstack
monteMatrix = np.zeros((monteCarlo+1, 4*AGwidth+2))
# matrix listing excitations used for alphas and gammas
excitations = np.array(alphaData[0, 0])
for n in range(AGwidth):
    excitations = np.append(excitations, alphaData[0, 2*n+1])

# the *drumroll* monte carlo
for i in range(len(observedSidebands)):

    # header info for the save file slice
    monteSlice = np.array(len(observedSidebands))
    # this inclusion is helpful but redundant, and is mostly padding
    monteSlice = np.append(monteSlice, monteCarlo)
    for j in range(AGwidth):
        monteSlice = np.append(monteSlice, alphaData[i+1, 2*j+1])
        monteSlice = np.append(monteSlice, alphaData[i+1, 2*j+2])
    for j in range(AGwidth):
        monteSlice = np.append(monteSlice, gammaData[i+1, 2*j+1])
        monteSlice = np.append(monteSlice, gammaData[i+1, 2*j+2])

    '''
    at this point monteSlice should be ordered as follows
    #sidebands|monteCarlo|alphavalue|alphaerror|...|gammavalue|gammaerror|...
    '''

    for m in range(monteCarlo):

        appendMatrix = np.array(m)
        appendMatrix = np.append(appendMatrix, alphaData[i+1, 0])

        alphas = np.array(alphaData[1, 0])
        # start alphas with the sideband being calculated
        for n in range(AGwidth):
            alphas = np.append(alphas, np.random.normal(alphaData[i+1, 2*n+1],
                                                        alphaData[i+1, 2*n+2]))
        # use 2n to skip the first element which is measured sideband
        # append elements into the empty array from the extracted array
        appendMatrix = np.append(appendMatrix, alphas[1:])
        '''
        appendMatrix should be formatted as follows at this point
        iteration#|alpha|...

        alphas should be as follows
        -1|excitation angles|
        sideband#|alpha|...
        '''
        # put recorded alphas for this iteration into the slice
        alphas = np.vstack((excitations, alphas))
        # stack alphas with excitation for extracting jones matrix

        # repeat process above with gammas
        gammas = np.array(gammaData[1, 0])
        for n in range(AGwidth):
            gammas = np.append(gammas, np.random.normal(gammaData[i+1, 2*n+1],
                                                        gammaData[i+1, 2*n+2]))
        appendMatrix = np.append(appendMatrix, gammas[1:])
        gammas = np.vstack((excitations, gammas))
        '''
        appendMatrix should be formatted as follows at this point
        iteration#|sideband#|alpha|...|gamma|...

        gammas should be as follows
        -1|excitation angles|
        sideband#|gamma|...
        '''
        # feed alpha and gamma arrays to the jones matrix function
        J = qwp.extractMatrices.findJ(alphas, gammas)
        # reshape from a 2d array into a 1d array
        J = np.reshape(J, -1)
        # cast parts of J to floats using np.imag and np.real, then append
        for n in range(len(J)):
            appendMatrix = np.append(appendMatrix, np.real(J[n]))
            appendMatrix = np.append(appendMatrix, np.imag(J[n]))
        '''
        appendMatrix should be formatted as follows at this point
        iteration#|sideband#|alpha|...|gamma|...|jonesReal|jonesImag|...
        '''
        monteSlice = np.vstack((monteSlice, appendMatrix))

    '''
    at this point monteSlice should be ordered as follows
    #sidebands|monteCarlo|alphavalue|alphaerror|...|gammavalue|gammaerror|...
    iteration#|sideband#|alpha|...|gamma|...|jonesReal|jonesImag|...
    '''
#                ||
#               \||/
#                \/
    '''
    for # iterations in range(monteCarlo)
    '''
    monteMatrix = np.dstack((monteMatrix, monteSlice))

# cut out zeros monteMatrix started with
monteMatrix = np.array(monteMatrix[:, :, 1:])

os.mkdir('./'+saveFolder)
np.save('monteArray', monteMatrix)


# and save the matrices to text
# qwp.extractMatrices.saveT(
# J, observedSidebands, "{}_JMatrix.txt".format(saveFileName))
