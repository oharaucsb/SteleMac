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
    plt.title('Sideband ' + str(int(observedSidebands[i])))

    # construct alpha histogram subplots
    for j in range(AGwidth):
        alphaMu = monteMatrix[0, 2+(j*2), i]
        alphaSigma = monteMatrix[0, 3+(j*2), i]
        plt.subplot(AGwidth, 3, (3*j+1))
        aCount, aBins, aIgnored = plt.hist(
            np.reshape(monteMatrix[1:, 2+j, i], -1),
            30, density=True)
        plt.plot(aBins, 1/(alphaSigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (aBins - alphaMu)**2 / (2 * alphaSigma**2)),
                 linewidth=2, color='r')
        plt.ylabel(excitations[j])
        if j == 0:
            plt.title('alphas')
        plt.yticks([])

    # construct gamma histogram subplots
    for j in range(AGwidth):
        gammaMu = monteMatrix[0, 10+(j*2), i]
        gammaSigma = monteMatrix[0, 11+(j*2), i]
        plt.subplot(AGwidth, 3, (3*j+2))
        aCount, aBins, aIgnored = plt.hist(
            np.reshape(monteMatrix[1:, 6+j, i], -1),
            30, density=True)
        plt.plot(aBins, 1/(gammaSigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (aBins - gammaMu)**2 / (2 * gammaSigma**2)),
                 linewidth=2, color='r')
        plt.ylabel(excitations[j])
        if j == 0:
            plt.title('gammas')
        plt.yticks([])

    # construct jones matrix xy axis subplots
    for j in range(4):
        plt.subplot(4, 3, 3*j+3)
        # do some magic here
        plt.ylabel(jones[j])
        if j == 0:
            plt.title('Jones')
        plt.yticks([])
    plt.tight_layout()
    plt.show()
