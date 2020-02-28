import Stele.ipg as pg
import numpy as np
import Stele as hsg
import Stele.QWPProcessing as qwp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import os
import scipy as sp


# object class to construct and hold monte carlo matrix, as well as analyse
class monteCarlo(object):

    # object members will be
    # name of folder things are saved and loaded
    folder_name = None
    # array in monte carlo format
    monteMatrix = None
    # total monte Carlo iterations per sideband
    nMonteCarlo = None
    # width of alpha and gamma measurements
    AGwidth = None
    # string of jones matrix indice names
    _jones = None
    # list of excitation numbers
    _excitations = None
    # array of sideband numbers
    _observedSidebands = None

    # initialization function for object, will follow one of 3 modes
    # initialization from raw input alphas and gammas
    # initialization from a saved monte carlo matrix
    # initialization from being pointed to a Fan Data file - this can be last
    def __init__(self, alphaData=None, gammaData=None,  nMonteCarlo=5000,
                 # TODO: remove observedSidebands and calculate it from matrix
                 folder_name=None, observedSidebands=None):

        # store folder name to self for either loading or saving
        self.folder_name = folder_name
        # construct jones matrix strings
        self._jones = ['xx', 'xy', 'yx', 'yy']
        # construct excitation matrix strings
        # TODO: calculate strings from algha/gamma data monte matrix line
        self._excitations = ['0', '-45', '-90', '45']

        # begin initialization from a passed alpha and gamma
        if (alphaData is not None) & (gammaData is not None):
            # save an array of sideband numbers
            self._observedSidebands
            # set the number of monte carlo iterations and record
            self.nMonteCarlo = nMonteCarlo
            # set how many different alpha or gamma values per matrix
            # in other words number of excitation angles used
            self.AGwidth = 0
            # begin monte carlo matrix with row of zeroes for subsequent Vstack
            self.monteMatrix = np.zeros((self.nMonteCarlo+1, 4*self.AGwidth+2))
            # harvest excitation numbers for use in alpha and gamma inputs
            excitations = np.array(alphaData[0, 0])
            for n in range(self.AGwidth):
                excitations = np.append(excitations, alphaData[0, 2*n+1])

            # the monte carlo creation
            for i in range(len(self._observedSidebands)):

                # header info for the save file slice
                monteSlice = np.array(len(self._observedSidebands))
                # this inclusion is helpful but redundant, is mostly padding
                monteSlice = np.append(monteSlice, self.nMonteCarlo)
                for j in range(self.AGwidth):
                    monteSlice = np.append(monteSlice, alphaData[i+1, 2*j+1])
                    monteSlice = np.append(monteSlice, alphaData[i+1, 2*j+2])
                for j in range(self.AGwidth):
                    monteSlice = np.append(monteSlice, gammaData[i+1, 2*j+1])
                    monteSlice = np.append(monteSlice, gammaData[i+1, 2*j+2])

                '''
at this point monteSlice should be ordered as follows
#sidebands|self.nMonteCarlo|alphavalue|alphaerror|...|gammavalue|gammaerror|...
                '''

                for m in range(self.nMonteCarlo):

                    appendMatrix = np.array(m)
                    appendMatrix = np.append(appendMatrix, alphaData[i+1, 0])

                    alphas = np.array(alphaData[1, 0])
                    # TODO: group alpha and gamma code to reduce complexity
                    # start alphas with the sideband being calculated
                    for n in range(self.AGwidth):
                        alphas = np.append(alphas, np.random.normal(
                            alphaData[i+1, 2*n+1],
                            alphaData[i+1, 2*n+2]))
                    # use 2n to skip the first element, measured sideband
                    # append elements to empty array from the extracted array
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
                    for n in range(self.AGwidth):
                        gammas = np.append(gammas, np.random.normal(
                            gammaData[i+1, 2*n+1],
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
                    # cast parts of J to floats using np.imag and np.real,
                    # then append
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
# TODO: remove self.nMonteCarlo column from Monte Carlo Matrix
#sidebands|self.nMonteCarlo|alphavalue|alphaerror|...|gammavalue|gammaerror|...
iteration#|sideband#|alpha|...|gamma|...|jonesReal|jonesImag|...
...
for # iterations in range(self.nMonteCarlo)
                '''
                self.monteMatrix = np.dstack((self.monteMatrix, monteSlice))

            # cut out zeros self.monteMatrix started with
            self.monteMatrix = np.array(self.monteMatrix[:, :, 1:])

            # if folder name is passed, save monte carlo matrix to that folder
            if self.folder_name is not None:
                # construct destination folder
                os.mkdir('./'+self.folder_name)
                # save array to that folder
                np.save('./'+self.folder_name+'monteArray', self.monteMatrix)

        # construct monte carlo from a save destination
        elif self.folder_name is not None:
            # loads a monte carlo matrix from folder location
            self.monteMatrix = np.load('./'+self.folder_name
                                       + '/monteArray.npy')
            # calculate self.AGwidth from monte carlo matrix loaded
            self.AGwidth = int(len(np.reshape(
                self.monteMatrix[0, 2:, 0], -1))/4)
            # pull monte carlo number from monte carlo matrix
            self.nMonteCarlo = int(len(self.monteMatrix[1:, 0, 0]))
            # pull observed sideband from monte carlo matrix
            self._observedSidebands = np.array(
                                    np.reshape(self.monteMatrix[1, 1, :], -1))

    # begin graphing functions
    # return histogram figure of alphas and gammas
    def AGHistogram(sidebands=None):
        # if i is a member of sidebands, graph, if sidebands  is none, graph
    # return figure of histogram of jones matrix
    # return figure of scatterplot of jones matrix
    # return figure of contour plot of jones matrix at standard deviation
    # return matrix of mu and sigma values organized
    # return figure of monte carlo run plots
    # save some combination of figures to self.folder_name
