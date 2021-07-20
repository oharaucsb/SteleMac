import numpy as np
import matplotlib.pyplot as plt
import Stele.analysis.analysis_hsg.helper_functions as help
import Stele.processing.processing_qwp.fan_compiler as fancomp
import Stele.analysis.analysis_hsg.collection_functions as hsgcoll
import Stele.processing.processing_qwp.extract_matrices as extmat


def J_T_proc(file, observedSidebands, crystalAngle, saveFileName,
             save_results=False, Plot=False, keepErrors=False):
    """
    This function will take a folder of polarimetry scans and produce
    DOP, alpha/gamma, J and T matrices, and Matrix Relations This is the same
    as fan_n_Tmat but doesn't create the fan itself. Otherwise creates pretty
    much everything one would need.

    :param file: String of folder name containing 4 polarimetry scans
    :param observedSidebands: np array of observed sidebands. Data will be
        cropped such that these sidebands are included in everything.
    :param crystalAngle: (Float) Angle of the sample from the 010 crystal face
    :saveFileName: Str of what you want to call the text files to be saved
    :save_results: Boolean controls if things are saved to txt files.
        Currently saves alpha, gamma, J matrix, T matrix, Fan, Matrix
        Relations, and DOP
    :Plot: Boolean controls if plots of matrix relations are displayed

    :return: alpha,gamma,J matrix, T matrix, Matrix Relations, DOP
    """

    # Make a list of folders for the fan data
    datasets = help.natural_glob(file, "*")

    # FanCompiler class keeps alpha and gamma data together for different pols
    fanData = fancomp.FanCompiler(observedSidebands, keepErrors=keepErrors)

    # Initialize arrays for DOP
    dopsize = (len(observedSidebands), 2*len(datasets)+1)
    doparr = np.zeros(dopsize)
    doparr[:, 0] = observedSidebands
    dophead = 'SB Order, '

    dataidx = 0  # Keeps track of data iteration
    for data in datasets:
        laserParams, rawData = hsgcoll.hsg_combine_qwp_sweep(data)
        try:
            _, fitDict = hsgcoll.proc_n_fit_qwp_data(
                rawData, vertAnaDir=False,
                laserParams=laserParams,
                wantedSbs=observedSidebands)
            # Add the new alpha and gamma sets to the fan class
        except IndexError:
            print('incorrect number of files in data folder ', data)
            print('proceeding without fourier analysis')
            _, fitDict = hsgcoll.proc_n_fit_qwp_data(
                rawData, vertAnaDir=False,
                laserParams=laserParams,
                wantedSbs=observedSidebands, fourier=False)
        # Add the new alpha and gamma sets to the fan class
        fanData.addSet(fitDict)

        # Get Stoke Parameters
        untrim_s0 = fitDict['S0']
        untrim_s1 = fitDict['S1']
        untrim_s2 = fitDict['S2']
        untrim_s3 = fitDict['S3']

        s0 = np.zeros((len(observedSidebands), 3))
        s1 = np.zeros((len(observedSidebands), 3))
        s2 = np.zeros((len(observedSidebands), 3))
        s3 = np.zeros((len(observedSidebands), 3))

        # Trims the Stoke Parameters down to remove any sidebands beyond
        # observedSidebands
        # This does mean that it won't calculate the DOP for any extra
        # sidebands, even if the software detected them.

        for nidx in np.arange(len(untrim_s0[:, 0])):
            n = untrim_s0[nidx, 0]
            if n in observedSidebands:
                nprime = np.where(observedSidebands == n)[0]
                s0[nprime, :] = untrim_s0[nidx, :]
                s1[nprime, :] = untrim_s1[nidx, :]
                s2[nprime, :] = untrim_s2[nidx, :]
                s3[nprime, :] = untrim_s3[nidx, :]
        # This actually calculates the DOP and DOP error
        dop = s0
        dop[:, 1] = (np.sqrt((s1[:, 1])**2+(s2[:, 1])**2 +
                     (s3[:, 1])**2)/s0[:, 1])
        dop[:, 2] = (np.sqrt((s1[:, 1]**2)*(s1[:, 2]**2)/(s0[:, 1]**2)/(
                     (s1[:, 1])**2+(s2[:, 1])**2+(s3[:, 1])**2) +
                     (s2[:, 1]**2)*(s2[:, 2]**2)/(s0[:, 1]**2)/(
                     (s1[:, 1])**2+(s2[:, 1])**2+(s3[:, 1])**2) +
                     (s3[:, 1]**2)*(s3[:, 2]**2)/(s0[:, 1]**2)/(
                     (s1[:, 1])**2+(s2[:, 1])**2+(s3[:, 1])**2)) +
                     ((s1[:, 1])**2+(s2[:, 1])**2+(s3[:, 1])**2) *
                     s0[:, 2]**2/s0[:, 1]**4)

        aglab = str((laserParams['lAlpha'], laserParams['lGamma']))
        dophead += aglab+' DOP, '
        dophead += aglab+' DOP Error, '
        doparr[:, 2*dataidx+1] = dop[:, 1]
        doparr[:, 2*dataidx+2] = dop[:, 2]
        dataidx += 1
    # Saves as txt file with columns, SB Order, 00 DOP and error, 45 DOP and
    # error, 90 DOP and error, -45 DOP and error

    # Building the fan compiler causes it to make  2D np arrays with the alpha
    # and gamma angles where each column is a different alpha_NIR excitation
    alphaData, gammaData, _ = fanData.build()

    # save the alpha and gamma results
    if save_results:
        fanData.buildAndSave(saveFileName + "_{}.txt")

    # Now we need to calculate the Jones matrices.
    if keepErrors:
        MatalphaData = alphaData[:, [0, 1, 3, 5, 7]]
        MatgammaData = gammaData[:, [0, 1, 3, 5, 7]]
    else:
        MatalphaData = alphaData
        MatgammaData = gammaData

    J = extmat.findJ(MatalphaData, MatgammaData)

    # Get the T matrix:
    T = extmat.makeT(J, crystalAngle)

    # and save the matrices
    if save_results:
        extmat.saveT(J, observedSidebands,
                     "{}_JMatrix.txt".format(saveFileName))
        extmat.saveT(T, observedSidebands,
                     "{}_TMatrix.txt".format(saveFileName))

    # And to look at the ratios of the T matrix directly:
    if Plot:
        figmag, axmag = plt.subplots()
        axmag.set_title("Magnitudes")
        axmag.plot(observedSidebands,
                   np.abs(T[0, 0, :]/T[1, 1, :]),
                   'o-', label="T++/T--")
        axmag.plot(observedSidebands,
                   np.abs(T[0, 1, :]/T[1, 0, :]),
                   'o-', label="T+-/T-+")
        axmag.legend()

        figang, axang = plt.subplots()
        axang.set_title("Angles")
        axang.plot(observedSidebands,
                   np.angle(T[0, 0, :]/T[1, 1, :], deg=True),
                   'o-', label="T++/T--")
        axang.plot(observedSidebands,
                   np.angle(T[0, 1, :]/T[1, 0, :], deg=True),
                   'o-', label="T+-/T-+")
        axang.legend()

        plt.show()

    # Ok this is sort of a quick way to get what I want for Origin plotting
    # the relevant T matrix values
    #
    # ToDo: Format the text files to fit with the standards of other txt files

    tmag = np.transpose(np.array([observedSidebands,
                                  np.abs(T[0, 0, :]/T[1, 1, :]),
                                  np.abs(T[0, 1, :]/T[1, 0, :])]))
    tang = np.transpose(np.array([observedSidebands,
                                  np.angle(T[0, 0, :]/T[1, 1, :], deg=True),
                                  np.angle(T[0, 1, :]/T[1, 0, :], deg=True)]))

    if save_results:
        np.savetxt(saveFileName + "_{}.txt".format("TmatrixMag"), tmag,
                   delimiter=',', header='SB Order, |T++/T--|, |T+-/T-+|')
        np.savetxt(saveFileName + "_{}.txt".format("TmatrixAng"), tang,
                   delimiter=',',
                   header='SB Order, Angle(T++/T--), Angle(T+-/T-+)')
        np.savetxt(saveFileName + "_{}.txt".format("DOP"), doparr,
                   delimiter=',', header=dophead)

    return alphaData, gammaData, J, T, tmag, tang, doparr
