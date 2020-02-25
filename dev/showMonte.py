import numpy as np
import matplotlib.pyplot as plt

AGwidth = 4
monteMatrix = np.load('monteArray')
# display results for each given sideband
for i in range(len(observedSidebands)):

    # plot alpha histogram
    # plt.subplot(AGwidth, 3, 1)
    # plt.title('Input Error Distribution')
    for j in range(AGwidth):
        alphaMu = alphaData[i+1, 2*j+1]
        alphaSigma = alphaData[i+1, 2*j+2]
        plt.subplot(AGwidth, 3, (3*j+1))
        aCount, aBins, aIgnored = plt.hist(monteMatrix[:, 2+j, i],
                                           30, density=True)
        plt.plot(aBins, 1/(alphaSigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (aBins - alphaMu)**2 / (2 * alphaSigma**2)),
                 linewidth=2, color='r')
    plt.ylabel('Alpha ' + str(int(excitations[j+1])))
    plt.yticks([])
