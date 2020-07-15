import matplotlib.pyplot as plt
# from matplotlib import patches
import numpy as np
import scipy as sp

loadFolder = 'theta9000'
monteMatrix = np.load('./' + loadFolder + '/monteArray.npy')

# calculate values necessary for future use of matrix


'''
while True:
    print("Input choice to display alpha and gamma histograms, [Y/N]")
    histChoice = input()
    if histChoice == 'Y' or 'N' or 'y' or 'n':
        break
    else:
        print("Improper input detected")

print("Input choice to display (s)catterplot, (h)istogram, or (n)either")
plotchoice = input()
'''

# TODO: Convert the following function into a comprehensive object
#           break separate graphing and output functions into member functions


def carloAnalysis(monteMatrix,
                  # show determines if relevant plots are displayed or not
                  # plotAGs determines if alpha & gamma histograms are plotted
                  show=False, plotAGs=False,
                  # plotConfidence toggles a monte carlo vs sigma values plot
                  # jonesPlotType toggles s=scatter, h=2D histogram, c=contour
                  plotConfidence=False, jonesPlotType='s',
                  # saveFolder determines output destination folder for graphs
                  # imageType determines saved image format of graphs
                  saveFolder='NONE', imageType='png'):

    # calculate values necessary for future use of matrix
    AGwidth = int(len(np.reshape(monteMatrix[0, 2:, 0], -1))/4)
    excitations = ['0', '-45', '-90', '45']
    jones = ['xx', 'xy', 'yx', 'yy']
    observedSidebands = np.array(np.reshape(monteMatrix[1, 1, :], -1))
    MuSigmaArray = np.zeros(AGwidth*4+1)

    # create initial figure
    # display results for each given sideband
    for i in range(len(observedSidebands)):
        # for i in range(1):
        arrayAppend = np.array(observedSidebands[i])
        # plot alpha histogram
        # plt.subplot(AGwidth, 3, 1)
        # create figure for subplots to go under
        fig = plt.figure()
        # if not a X1, X2, X3 order band or if a X1X sideband eg 12th
        if ((int(observedSidebands[i]) % 10 >= 4) or
           (int(observedSidebands[i]) % 10 == 0) or
           (np.floor(int(observedSidebands[i]) / 10) % 10 == 1)):
            fig.suptitle(str(int(observedSidebands[i])) + 'th order sideband')
        else:
            # sidebands should always be even I believe but just in case
            suffix = ['st', 'nd', 'rd']
            fig.suptitle(str(int(observedSidebands[i])) +
                         suffix[int(observedSidebands[i]) % 10 - 1]
                         + ' order sideband')

        if (((i == 0) or (i == (len(observedSidebands)-1))) and
           plotConfidence):

            # create figure onto which plots will be created
            conPlot = fig

            # generating data points to be plotted
            # create array of all monte carl numbers cast to integers
            carlo = np.array((monteMatrix[1:, 0, i]).astype(int))
            # find even distance between indices excluding 0th
            # such that index*100 is final indice address
            index = ((len(carlo)-1)/100)
            # begin X array with 1st element, need 2 values for a sigma
            X = np.array(carlo[0+int(np.round(index))])
            # x axis is monte carlo number, sampled 100 times evenly
            # goes from 0 to 98
            for n in range(99):
                # create appendments onto X of evenly spaced monte values
                X = np.append(X, carlo[int(np.round((n+2)*index))])

            # y axis is sigma value for that number of monte carlo runs
            # assign sigma value of monte carlo 0
            # xyReal: 12, xyImag: 13, yyReal: 16 , yyImag: 17
            jonesIndice = [12, 13, 16, 17]
            for jI in range(int(len(jonesIndice))):
                # start Y array with standard deviation from 0th to X[0]
                Y = np.array(np.std(monteMatrix[1:X[0],
                             jonesIndice[jI], i]))
                for n in range(99):
                    # fill in the other 99 elements from X[1] to x[99]
                    Y = np.append(Y, np.std(monteMatrix[1:X[n+1],
                                  jonesIndice[jI], i]))
                # add subplot to figure
                sbp = conPlot.add_subplot(int(np.round(len(jonesIndice)/2)),
                                          2, jI+1)
                sbp.plot(X[1:], Y[1:])
            conPlot.show()
            plt.show()
            # curve fitting the plot
            # Xnew = np.linespace(X.min(), X.max(), 300)
            # spl = sp.make_interp_spline(X, Y, Xnew)
            #

        # construct alpha histogram subplots
        for j in range(AGwidth):
            alphaMu = monteMatrix[0, 2+(j*2), i]
            alphaSigma = monteMatrix[0, 3+(j*2), i]
            sbp = fig.add_subplot(AGwidth, 3, (3*j+1))
            sbp.set_ylabel(excitations[j])
            if j == 0:
                sbp.set_title('alphas')
            sbp.set_yticks([])
            aCount, aBins, aIgnored = sbp.hist(
                np.reshape(monteMatrix[1:, 2+j, i], -1),
                30, density=True)
            sbp.plot(aBins, 1/(alphaSigma * np.sqrt(2 * np.pi)) *
                     np.exp(- (aBins - alphaMu)**2 / (2 * alphaSigma**2)),
                     linewidth=2, color='r')

        # construct gamma histogram subplots
        for j in range(AGwidth):
            gammaMu = monteMatrix[0, 10+(j*2), i]
            gammaSigma = monteMatrix[0, 11+(j*2), i]
            sbp = fig.add_subplot(AGwidth, 3, (3*j+2))
            sbp.set_ylabel(excitations[j])
            if j == 0:
                sbp.set_title('gammas')
            sbp.set_yticks([])
            aCount, aBins, aIgnored = sbp.hist(
                np.reshape(monteMatrix[1:, 6+j, i], -1),
                30, density=True)
            sbp.plot(aBins, 1/(gammaSigma * np.sqrt(2 * np.pi)) *
                     np.exp(- (aBins - gammaMu)**2 / (2 * gammaSigma**2)),
                     linewidth=2, color='r')

        # construct jones matrix xy axis scatterplot subplots
        for j in range(4):
            sbp = fig.add_subplot(4, 3, 3*j+3)
            sbp2 = sbp.twinx()
            sbp2.set_ylabel(jones[j])
            if j == 0:
                sbp2.set_title('Jones')
                sbp.set_ylabel('Imaginary')
                sbp.set_xlabel('Real')
                sbp.set_yticks([])
                sbp.set_xticks([])
            sbp2.set_yticks([])
            # do some magic here
            # real number mean & sigma
            jrMu = np.mean(monteMatrix[1:, 10+(2*j), i])
            arrayAppend = np.append(arrayAppend, jrMu)
            jrSigma = np.std(monteMatrix[1:, 10+(2*j), i])
            arrayAppend = np.append(arrayAppend, jrSigma)
            # imaginary number mean & sigma
            jiMu = np.mean(monteMatrix[1:, 11+(2*j), i])
            arrayAppend = np.append(arrayAppend, jiMu)
            jiSigma = np.std(monteMatrix[1:, 11+(2*j), i])
            arrayAppend = np.append(arrayAppend, jiSigma)
            '''
            # display mu and sigma values for given jones on sideband slice
            print(jones[j]+' jreal: $mu$:' + str(jrMu)
                + ' $sigma$:' + str(jrSigma)
                + ' jimag: $mu$ '+str(jiMu)
                + ' $sigma$ ' + str(jiSigma))
            '''
            # '''
            # scatterplot method
            sbp.scatter(monteMatrix[1:, 10+(2*j), i],
                        monteMatrix[1:, 11+(2*j), i],
                        s=1,
                        marker='.')

            # single point plot of mean values
            sbp.scatter(jrMu, jiMu, c='r', marker="1")
            # save scatter plot for that order within folder
            fig.savefig(('./' + loadFolder + '/order_' +
                        str(int(observedSidebands[i]))
                        + '_scatterplot.png'))
            '''

            # 2D distribution would be ellipse with semimajor axis as Sigmas
            # confidence_ellipse(jrSigma, jiSigma, sbp)
            # twoDSigma = patches.Ellipse(xy=(jrMu, jiMu),
            #                             width=jrSigma, height=jiSigma,
            #                             edgecolor='r')
            # sbp.add_patch(twoDSigma)

            # 2d histogram method
            sbp.hist2d(monteMatrix[1:, 10+(2*j), i],
                       monteMatrix[1:, 11+(2*j), i],
                       20)
            # save histogram plot to folder
            fig.savefig('./' + loadFolder + '/order_' +
                        str(observedSidebands[i])
                        + '_histogram')
            '''

            # fit plot layout and display
        MuSigmaArray = np.vstack((MuSigmaArray, arrayAppend))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        if show is True:
            fig.show()
    if saveFolder != 'NONE':
        # save output text matrix of mu and sigma values
        np.savetxt('./' + saveFolder + '/MuSigmaArray', MuSigmaArray[1:])
    # plt.show()


carloAnalysis(monteMatrix, saveFolder=loadFolder, plotConfidence=True)
