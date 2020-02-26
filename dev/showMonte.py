import matplotlib.pyplot as plt
import numpy as np


monteMatrix = np.load('monteArray.npy')

# calculate values necessary for future use of matrix
AGwidth = int(len(np.reshape(monteMatrix[0, 2:, 0], -1))/4)
excitations = ['0', '-45', '-90', '45']
jones = ['xx', 'xy', 'yx', 'yy']
observedSidebands = np.array(np.reshape(monteMatrix[1, 1, :], -1))

# display results for each given sideband
# for i in range(len(observedSidebands)):
for i in range(1):

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
        # sidebands should always be even I believe but in case of other uses
        suffix = ['st', 'nd', 'rd']
        fig.suptitle(str(int(observedSidebands[i])) +
                     suffix[int(observedSidebands[i]) % 10 - 1]
                     + ' order sideband')

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
        sbp2.set_yticks([])
        sbp.set_ylabel('Imaginary')
        # do some magic here
        sbp.hist2d(monteMatrix[1:, 10+(2*j), i],
                   monteMatrix[1:, 11+(2*j), i],
                   20)
# sbp.scatter(monteMatrix[1:, 10+(2*j), i], monteMatrix[1:, 11+(2*j), i])
        # end of magic

    # fit plot layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
