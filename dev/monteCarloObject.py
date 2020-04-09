import Stele.ipg as pg
import numpy as np
import Stele as hsg
import Stele.QWPProcessing as qwp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import os
import scipy as sp


# TODO: add a marker on graphs showing jones matrix values of mu alpha gamma

# TODO: update where needed to account for additional deadslice of mu average

# TODO: remove #sidebands from header and monte carlo number from slices
# #sidebands can be found by Z traversing 3D matrix
# carlo number can just be found by counting if needed

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
    # list of excitation numbers for alpha
    _alphaExcitations = None
    # list of excitation numbers for alpha
    _gammaExcitations = None
    # array of sideband numbers
    _observedSidebands = None
    # number of dead rows before monte carlo Data
    _filler = 2

    # helper functions
    # return proper title for given sideband index
    def __orderTitle(self, i):
        if ((int(self._observedSidebands[i]) % 10 >= 4) or
           (int(self._observedSidebands[i]) % 10 == 0) or
           (np.floor(int(self._observedSidebands[i]) / 10) % 10 == 1)):
            return (str(int(self._observedSidebands[i]))+'th order sideband')
        else:
            # sidebands should always be even I believe but just in case
            suffix = ['st', 'nd', 'rd']
            return(str(int(self._observedSidebands[i])) +
                   suffix[int(self._observedSidebands[i]) % 10 - 1]
                   + ' order sideband')

    def __stringJ(self, alphas, gammas):
        # feed alpha and gamma arrays to the jones matrix function
        J = qwp.extractMatrices.findJ(alphas, gammas)
        # reshape from a 2d array into a 1d array
        J = np.reshape(J, -1)
        # cast parts of J to floats using np.imag and np.real,
        # then append
        appendMatrix = np.array(0)
        for n in range(len(J)):
            appendMatrix = np.append(appendMatrix, np.real(J[n]))
            appendMatrix = np.append(appendMatrix, np.imag(J[n]))
        return appendMatrix[1:]

    # initialization function for object, will follow one of 3 modes
    # initialization from raw input alphas and gammas
    # initialization from a saved monte carlo matrix
    # initialization from being pointed to a Fan Data file - this can be last
    def __init__(
         self, alphaData=None, gammaData=None,  nMonteCarlo=5000,
         # TODO: remove observedSidebands and calculate it from matrix
         # store folder name to self for either loading or saving
         # observedSidebands is a twople of the first & last sideband numbers
         folder_name=None, observedSidebands=None):
        self.folder_name = folder_name
        # construct jones matrix strings
        self._jones = ['xx', 'xy', 'yx', 'yy']
        # construct excitation matrix strings
        # TODO: calculate strings from algha/gamma data monte matrix line
        self._alphaExcitations = ['0', '-45', '-90', '45']
        self._gammaExcitations = ['0', '0', '0', '0']
        # begin initialization from a passed alpha and gamma
        if (alphaData is not None) & (gammaData is not None):
            # save an array of sideband numbers
            self._observedSidebands = np.arange(observedSidebands[0],
                                                (observedSidebands[1]+1),
                                                2)
            # set the number of monte carlo iterations and record
            self.nMonteCarlo = nMonteCarlo
            # set how many different alpha or gamma values per matrix
            # in other words number of excitation angles used
            self.AGwidth = 4
            # begin monte carlo matrix with row of zeroes for subsequent Vstack
            self.monteMatrix = np.zeros(((self.nMonteCarlo+self._filler),
                                         (4*self.AGwidth+2)))
            # harvest excitation numbers for use in alpha and gamma inputs
            alphaExcitations = np.array(alphaData[0, 0])
            gammaExcitations = np.array(gammaData[0, 0])
            for n in range(self.AGwidth):
                alphaExcitations = np.append(alphaExcitations,
                                             alphaData[0, 2*n+1])
                gammaExcitations = np.append(gammaExcitations,
                                             gammaData[0, 2*n+1])

            # the monte carlo creation
            for i in range(len(self._observedSidebands)):

                # header info for the save file slice
                monteSlice = np.array(len(self._observedSidebands))
                # this inclusion is helpful but redundant, is mostly padding
                monteSlice = np.append(monteSlice, self.nMonteCarlo)
                monteSlice = np.append(monteSlice, alphaData[(1+i), 1:])
                monteSlice = np.append(monteSlice, gammaData[(1+i), 1:])

# at this point monteSlice should be ordered as follows
# #sidebands|self.nMonteCarlo|alphavalue|alphaerror|..|gammavalue|gammaerror|..

# create row for mu alpha sigma output to be used in comparison later
                appendMatrix = np.array(-2)
                appendMatrix = np.append(appendMatrix, alphaData[1, 0])
                alphas = np.array(alphaData[1, 0])
                gammas = np.array(gammaData[1, 0])
                for i in range(self.AGwidth):
                    alphas = np.append(alphas,
                                       monteSlice[2+2*i])
                    gammas = np.append(gammas,
                                       monteSlice[10+2*i])
                appendMatrix = np.append(appendMatrix, alphas[1:])
                appendMatrix = np.append(appendMatrix, gammas[1:])
                alphas = np.vstack((alphaExcitations, alphas))
                gammas = np.vstack((gammaExcitations, alphas))
# find jones values of mu and append them
                appendMatrix = np.append(appendMatrix,
                                         self.__stringJ(alphas, gammas))
                monteSlice = np.vstack((monteSlice, appendMatrix))

                '''
append madtrix should be structured as follows based on mu values
-2|sideband#|alpha|...|gamma|...|jonesReal|jonesImag|...
                '''

                for m in range(self.nMonteCarlo):

                    appendMatrix = np.array(m)
                    appendMatrix = np.append(appendMatrix, alphaData[i+1, 0])

                    alphas = np.array(alphaData[1, 0])
                    gammas = np.array(gammaData[1, 0])
                    # TODO: group alpha and gamma code to reduce complexity
                    # start alphas with the sideband being calculated
                    for n in range(self.AGwidth):
                        alphas = np.append(alphas, np.random.normal(
                            alphaData[i+1, 2*n+1],
                            alphaData[i+1, 2*n+2]))
                        gammas = np.append(gammas, np.random.normal(
                            gammaData[i+1, 2*n+1],
                            gammaData[i+1, 2*n+2]))
                    # use 2n to skip the first element, measured sideband
                    # append elements to empty array from the extracted array
                    appendMatrix = np.append(appendMatrix, alphas[1:])
                    appendMatrix = np.append(appendMatrix, gammas[1:])
# stack alphas with excitation for extracting jones matrix, repeat for gammas
                    alphas = np.vstack((alphaExcitations, alphas))
                    gammas = np.vstack((gammaExcitations, gammas))
                    '''
appendMatrix should be formatted as follows at this point
iteration#|alpha|...

alphas should be as follows
-1|excitation angles|
sideband#|alpha|...

gammas should be as follows
-1|excitation angles|
sideband#|gamma|...
                    '''
                    # append jones values calculated from alphas and gammas
                    appendMatrix = np.append(appendMatrix,
                                             self.__stringJ(alphas, gammas))

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
                # TODO: use try for folder construction in event it exists
                # construct destination folder
                os.mkdir('./'+self.folder_name)
                # TODO: test logic for overwriting an existing array
                # save array to that folder
                np.save('./'+self.folder_name+'/monteArray', self.monteMatrix)

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
    # return histogram array of plt.figure containing alphas and gammas
    def agHistogram(self, sidebands=None):
        figArray = np.array(plt.figur())
        # if i is a member of sidebands, or if sidebands  is none, graph
        for i in range(len(self._observedSidebands)):
            if (float(i) in sidebands) or (sidebands is None):
                # do histogram things
                fig = plt.figure()
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
                fig.suptitle(self.__orderTitle(i))
                for j in range(self.AGwidth):
                    alphaMu = self.monteMatrix[0, 2+(j*2), i]
                    alphaSigma = self.monteMatrix[0, 3+(j*2), i]
                    sbp = fig.add_subplot(self.AGwidth, 3, (3*j+1))
                    sbp.set_ylabel(self._alphaExcitations[j])
                    if j == 0:
                        sbp.set_title('alphas')
                    sbp.set_yticks([])
                    aCount, aBins, aIgnored = sbp.hist(
                        np.reshape(self.monteMatrix[1:, 2+j, i], -1),
                        30, density=True)
                    sbp.plot(aBins, 1/(alphaSigma * np.sqrt(2 * np.pi)) *
                             np.exp(- (aBins - alphaMu)**2 /
                             (2 * alphaSigma**2)), linewidth=2, color='r')

                # construct gamma histogram subplots
                for j in range(self.AGwidth):
                    gammaMu = self.monteMatrix[0, 10+(j*2), i]
                    gammaSigma = self.monteMatrix[0, 11+(j*2), i]
                    sbp = fig.add_subplot(self.AGwidth, 3, (3*j+2))
                    sbp.set_ylabel(self._alphaExcitations[j])
                    if j == 0:
                        sbp.set_title('gammas')
                    sbp.set_yticks([])
                    aCount, aBins, aIgnored = sbp.hist(
                        np.reshape(self.monteMatrix[1:, 6+j, i], -1),
                        30, density=True)
                    sbp.plot(aBins, 1/(gammaSigma * np.sqrt(2 * np.pi)) *
                             np.exp(- (aBins - gammaMu)**2 /
                             (2 * gammaSigma**2)), linewidth=2, color='r')

                # append constructed figure to array
                figArray = np.append(figArray, fig)

        figArray = np.array(figArray[1:])
        return figArray

    # return plt.figure array of histogram of jones matrix
    def JonesHistogram(self, sidebands=None):
        figArray = np.array(plt.figur())
        # if i is a member of sidebands, or if sidebands  is none, graph
        for i in range(len(self._observedSidebands)):
            if (float(i) in sidebands) or (sidebands is None):
                fig = plt.figure()
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
                fig.suptitle(self.__orderTitle(i))
                # TODO: jones histogram things

    # return plt.figure array of scatterplot of jones matrix
    def JonesScatter(self, sidebands=None):
        figArray = np.array(plt.figur())
        # if i is a member of sidebands, or if sidebands  is none, graph
        for i in range(len(self._observedSidebands)):
            if (float(i) in sidebands) or (sidebands is None):
                fig = plt.figure()
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
                fig.suptitle(self.__orderTitle(i))
                # TODO: jones scatterplot things
                for j in range(4):
                    sbp = fig.add_subplot(4, 3, 3*j+3)
                    sbp2 = sbp.twinx()
                    sbp2.set_ylabel(self._jones[j])
                    if j == 0:
                        sbp2.set_title('Jones')
                        sbp.set_ylabel('Imaginary')
                        sbp.set_xlabel('Real')
                        sbp.set_yticks([])
                        sbp.set_xticks([])
                    sbp2.set_yticks([])
                    # do some magic here
                    # scatterplot creation
                    sbp.scatter(self.monteMatrix[self._filler:, 10+(2*j), i],
                                self.monteMatrix[self._filler:, 11+(2*j), i],
                                s=1, marker='.')

                    # real & imaginary number means
                    jrMu = np.mean(self.monteMatrix[self._filler:,
                                                    10+(2*j), i])
                    jiMu = np.mean(self.monteMatrix[self._filler:,
                                                    11+(2*j), i])
                    # single point plot of mean values
                    sbp.scatter(jrMu, jiMu, c='r', marker="1")
                    # single point plot of mu matrix
                    sbp.scatter(self.monteMatrix[self._filler-1, 10+(2*j), i],
                                self.monteMatrix[self._filler-1, 11+(2*j), i],
                                c='g', marker="1")
                figArray = np.append(figArray, fig)
        return figArray[1:]

    # return figure array of contour plot of jones matrix at standard deviation
    def JonesContour(self, sidebands=None):
        figArray = np.array(plt.figur())
        # if i is a member of sidebands, or if sidebands  is none, graph
        for i in range(len(self._observedSidebands)):
            if (float(i) in sidebands) or (sidebands is None):
                fig = plt.figure()
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
                fig.suptitle(self.__orderTitle(i))
                # TODO: do jones countourplot things

    # return 2D array of mu and sigma values organized
    # sideband#|xxjrMu|xxjrSigma|xxjiMu|xxjiSigma|...
    def JonesSigmaMu(self, sidebands=None):
        returnStack = np.zeros(self.AGwidth*4+1)
        # if i is a member of sidebands, or if sidebands  is none, graph
        for i in range(len(self._observedSidebands)):
            if (float(i) in sidebands) or (sidebands is None):
                # convert in order to store in array with other floats
                arrayAppend = np.array(float(self._observedSidebands[i]))
                for j in range(4):
                    arrayAppend = np.append(
                      arrayAppend, np.mean(self.monteMatrix[1:, 10+(2*j), i]))
                    arrayAppend = np.append(
                      arrayAppend, np.std(self.monteMatrix[1:, 10+(2*j), i]))
                    # imaginary number mean & sigma
                    arrayAppend = np.append(
                      arrayAppend, np.mean(self.monteMatrix[1:, 11+(2*j), i]))
                    arrayAppend = np.append(
                      arrayAppend, np.std(self.monteMatrix[1:, 11+(2*j), i]))
                # vstack the arrays of sigma and Mus for the jones array
                returnStack = np.vstack(returnStack, arrayAppend)
        # strip off initialization zeros and return
        return returnStack[1:, :]

    # return figure of monte carlo run plots

    # save some combination of figures to self.folder_name
