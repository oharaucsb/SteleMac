import numpy as np
import Stele.ipg as pg
from scipy.optimize import minimize
from PyQt5 import QtCore, QtWidgets
from Stele import newhsganalysis
from Stele import JonesVector as JV
from .create_fan_diagram import createFan

newhsganalysis.plt = pg
np.set_printoptions(linewidth=400)


# This attempt is going to try and reuse the code that was used for the DBR
# paper.  It uses some system of equations that come from doing the matrix
# multipliations out and stuff

totalcount = 0


def unflattenJ(jVec):
    j = (jVec[:3]+1j*jVec[3:])
    j = np.append([1], j)
    return j.reshape(2, 2)


def solver(r, J):
    global totalcount
    totalcount += 1
    Jxy, Jyx, Jyy = (J[:3]+1j*J[3:])
    nir, sb = r
    eN = np.exp(1j*np.deg2rad(nir.delta))
    eH = np.exp(-1j*np.deg2rad(sb.delta))
    cotH = 1./np.tan(np.deg2rad(sb.phi))
    cN = np.cos(np.deg2rad(nir.phi))
    sN = np.sin(np.deg2rad(nir.phi))

    return cotH*eH*(Jyx*cN + Jyy*sN*eN)-cN-Jxy*sN*eN


def findJBad(obj):
    mod = QtWidgets.QApplication.keyboardModifiers()
    if mod & QtCore.Qt.ShiftModifier:
        print("Skipping")
        return
    alphaFullDatas = obj.alphaFullDatas
    alphaSBs = obj.sbs
    alphaAlphaNIRS = obj.nirAlphas
    gammaSBs = obj.sbs
    gammaAlphaNIRS = obj.nirAlphas
    sbs = obj.sbs
    nirAlphas = obj.nirAlphas
    cboxGroup = obj.cboxGroup
    sbGetter = obj.sbGetter
    palp = obj.palp
    pgam = obj.pgam
    intRatioCurve = obj.intRatioCurve

    outputAlphaData = np.empty(
        (alphaSBs.shape[0]+1, alphaAlphaNIRS.shape[0]+1)) * np.nan
    outputAlphaData[1:, 0] = alphaSBs
    outputAlphaData[0, 1:] = alphaAlphaNIRS

    outputGammaData = np.empty(
        (gammaSBs.shape[0] + 1, gammaAlphaNIRS.shape[0] + 1)) * np.nan
    outputGammaData[1:, 0] = gammaSBs
    outputGammaData[0, 1:] = gammaAlphaNIRS
    outputJMatrix = np.empty((len(sbs), 9))

    wantNIRAlphas = [ii for ii in nirAlphas if
                     cboxGroup.button(int(ii)).isChecked()]
    wantNIRIndices = np.array(
        [nirAlphas.tolist().index(ii) for ii in wantNIRAlphas])

    for idx, sb in enumerate(sbs):
        als, gms = zip(*[sbGetter(sb, ii) for ii in wantNIRAlphas])
        sbJones = JV(alpha=als, gamma=gms)
        nirJones = JV(alpha=wantNIRAlphas, gamma=0)

        costfunc = lambda jmatrix: np.linalg.norm(
            solver([nirJones, sbJones], jmatrix))

        p = minimize(costfunc, x0=np.ones(6))
        J = unflattenJ(p.x)

        # FOR SHOWING EVERYTHING
        nirJones = JV(alpha=nirAlphas, gamma=0)
        nirJones.apply_transformation(J)

        outputAlphaData[idx+1, 1:] = nirJones.alpha
        outputGammaData[idx+1, 1:] = nirJones.gamma

        np.set_printoptions(
            formatter={"float_kind": lambda x: "{: 6.2f}".format(x)})
        outputJMatrix[idx] = np.array(
            [sb, 1] + p.x[:3].tolist() + [0] + p.x[3:].tolist())

    palp.setImage(outputAlphaData[1:, 1:])
    palp.setLevels(-90, 90)
    pgam.setImage(outputGammaData[1:, 1:])
    pgam.setLevels(-45, 45)
    palp.imageItem.render()

    intRatioCurve.setData(alphaSBs, np.sqrt(
        np.abs(outputJMatrix[:, 4]**2+1j*outputJMatrix[:, -1]**2))
                          )

    obj.outputJMatrix = outputJMatrix


def updateTCurves():
    try:
        J54 = mainwid54.outputJMatrix
        Jm3 = mainwidm3.outputJMatrix
    except (AttributeError, NameError):
        return

    T54 = makeT(J54, 54)
    Tm3 = makeT(Jm3, -3)

    mainwid54.TMatrix = T54
    mainwidm3.TMatrix = Tm3

    Tm3pm = 180 / 3.14159 * np.angle(Tm3[0, 1, :] / Tm3[1, 0, :])
    T54pm = 180 / 3.14159 * np.angle(T54[0, 1, :] / T54[1, 0, :])

    print("T-3 +- {}".format(Tm3pm[:5].mean()))
    print("T54 +- {}".format(T54pm[:5].mean()))
    m3TpmTmp.setData(sbsm3, Tm3pm)
    p54TpmTmp.setData(sbsm3, T54pm)
    diffTpmTmp.setData(sbsm3, Tm3pm-T54pm)

    Tm3pp = 180 / 3.14159 * np.angle(Tm3[0, 0, :] / Tm3[1, 1, :])
    T54pp = 180 / 3.14159 * np.angle(T54[0, 0, :] / T54[1, 1, :])
    m3TppTmm.setData(sbsm3, Tm3pp)
    p54TppTmm.setData(sbsm3, T54pp)

    Tm3pp = np.abs(Tm3[0, 0, :])
    Tm3mm = np.abs(Tm3[1, 1, :])
    T54pp = np.abs(T54[0, 0, :])
    T54mm = np.abs(T54[1, 1, :])
    m3TppMag.setData(sbsm3, Tm3pp)
    m3TmmMag.setData(sbsm3, Tm3mm)
    p54TppMag.setData(sbsm3, T54pp)
    p54TmmMag.setData(sbsm3, T54mm)


def makeInteractiveFanWidget(
        compiledAlpha, compiledGamma, crystalOrientation, intensityData=[],
        plotFanNIRs=np.arange(-90, 90, 5), sliceInExp=True,
        calculationCallback=lambda J, T: T, **kwargs):
    """

    :param compiledAlpha: saved files created from a FanCompiler.buildAndSave()
    :param compiledGamma:
    :param calculationCallback: function to be called when T/J matrices are
        recalculated
    :return:
    """
    NMax = kwargs.get("NMax", 12)

    nirAlphas = compiledAlpha[0, 1:]
    sbs = compiledAlpha[1:, 0]
    print(f"Found sbs: {sbs}")
    print(f"found nira: {nirAlphas}")
    alphaData = compiledAlpha[1:, 1:]
    gammaData = compiledGamma[1:, 1:]

    mainwid = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    cboxesLayout = QtWidgets.QGridLayout()
    cboxGroup = QtWidgets.QButtonGroup()
    cboxGroup.setExclusive(False)

    for idx, ang in enumerate(nirAlphas):
        cb = QtWidgets.QCheckBox(str(ang))
        cb.setChecked(True)
        cboxGroup.addButton(cb, id=int(ang))
        cboxesLayout.addWidget(cb, idx // 12, idx % 12)

    layout.addLayout(cboxesLayout)
    tabWid = QtWidgets.QTabWidget(mainwid)

    palp, pgam = createFan(plotFanNIRs, sbs)


    def updateJ():

        mod = QtWidgets.QApplication.keyboardModifiers()
        if mod & QtCore.Qt.ShiftModifier:
            print("Skipping")
            return
        # Which NIR alpha angles you've selected to include in the J matrix
        # calculation
        wantNIRAlphas = [ii for ii in nirAlphas if
                         cboxGroup.button(int(ii)).isChecked()]
        # And converting those to indices. Include the 0th order to include
        # the SB
        wantNIRIndices = \
            [0] + [nirAlphas.tolist().index(ii)+1 for ii in wantNIRAlphas]

        toFitAlphas = compiledAlpha[:, wantNIRIndices]
        toFitGammas = compiledGamma[:, wantNIRIndices]

        J = findJ(toFitAlphas, toFitGammas)
        reconstructedAlpha, reconstructedGamma = jonesToFans(
            sbs, J, wantNIR=plotFanNIRs)

        if sliceInExp:
            for idx, _ in enumerate(nirAlphas):

                niralpha = compiledAlpha[0, idx+1]
                # finite gamma don't make
                # sense in this plot
                if np.abs(compiledGamma[0, idx+1]) > 1:
                    continue
                try:
                    reconstructedAlpha[
                        1:,
                        np.argwhere(
                            reconstructedAlpha[0, :].astype(int)
                            == niralpha)[0][0]
                        ] = compiledAlpha[1:, idx+1]

                    reconstructedGamma[
                        1:,
                        np.argwhere(
                            reconstructedGamma[0, :].astype(int)
                            == niralpha)[0][0]
                        ] = compiledGamma[1:, idx+1]
                except IndexError:
                    print("Warning! Unable to slice in NIR alpha = {}".format(
                        niralpha))

        palp.setImage(reconstructedAlpha[1:, 1:])
        pgam.setImage(reconstructedGamma[1:, 1:])

        palp.setLevels(-90, 90)
        pgam.setLevels(-45, 45)


        T = makeT(J, crystalOrientation)



        TppPolar.setData(
            sbs[:NMax]+np.abs(T[0, 0, :NMax]), np.angle(
                T[0, 0, :NMax], deg=False)
            )
        TpmPolar.setData(
            sbs[:NMax]+np.abs(T[0, 1, :NMax]), np.angle(
                T[0, 1, :NMax], deg=False)
            )
        TmpPolar.setData(
            sbs[:NMax]+np.abs(T[1, 0, :NMax]), np.angle(
                T[1, 0, :NMax], deg=False)
            )
        TmmPolar.setData(
            sbs[:NMax]+np.abs(T[1, 1, :NMax]), np.angle(
                T[1, 1, :NMax], deg=False)
            )

        TppLinear.setData(sbs[:NMax], np.abs(T[0, 0, :NMax]))
        TppALinear.setData(sbs[:NMax], np.angle(T[0, 0, :NMax], deg=True))
        TpmLinear.setData(sbs[:NMax], np.abs(T[0, 1, :NMax]))
        TpmALinear.setData(sbs[:NMax], np.angle(T[0, 1, :NMax], deg=True))
        TmpLinear.setData(sbs[:NMax], np.abs(T[1, 0, :NMax]))
        TmpALinear.setData(sbs[:NMax], np.angle(T[1, 0, :NMax], deg=True))
        TmmLinear.setData(sbs[:NMax], np.abs(T[1, 1, :NMax]))
        TmmALinear.setData(sbs[:NMax], np.angle(T[1, 1, :NMax], deg=True))

        TppoTmmLinear.setData(
            sbs[:NMax], np.abs(T[0, 0, :NMax] / T[1, 1, :NMax]))
        TppoTmmALinear.setData(
            sbs[:NMax], np.angle(
                T[0, 0, :NMax] / T[1, 1, :NMax], deg=True)
            )

        TpmoTmpLinear.setData(
            sbs[:NMax], np.abs(T[0, 1, :NMax] / T[1, 0, :NMax]))
        TpmoTmpALinear.setData(
            sbs[:NMax], np.angle(T[0, 1, :NMax] / T[1, 0, :NMax], deg=True))

        calculationCallback(J, T)
    # need to keep a reference
    mainwid.updateJ = updateJ

    tabWid.addTab(palp, "Fan Diagram")

    TPolars = makePolarPlot()
    pg.legend()
    tabWid.addTab(TPolars, "T Matrices (Polar)")

    TppPolar = polarPlot('ko-', name="T++")
    TpmPolar = polarPlot('ro-', name="T+-")
    TmpPolar = polarPlot('bo-', name="T-+")
    TmmPolar = polarPlot('mo-', name="T--")

    double = pg.DoubleYPlot()
    TLinears = pg.PlotContainerWindow(plotWidget=double)
    TLinears.addLegend()
    TppLinear = TLinears.plot('ko-', name="T++")
    TpmLinear = TLinears.plot('ro-', name="T+-")
    TmpLinear = TLinears.plot('bo-', name="T-+")
    TmmLinear = TLinears.plot('mo-', name="T--")

    TppALinear = TLinears.plot('ko--', name="T++")
    double.p2.addItem(TppALinear)
    TpmALinear = TLinears.plot('ro--', name="T+-")
    double.p2.addItem(TpmALinear)
    TmpALinear = TLinears.plot('bo--', name="T-+")
    double.p2.addItem(TmpALinear)
    TmmALinear = TLinears.plot('mo--', name="T--")
    double.p2.addItem(TmmALinear)

    tabWid.addTab(TLinears, "T Matrices (Linear)")

    double2 = pg.DoubleYPlot()
    TLinears2 = pg.PlotContainerWindow(plotWidget=double2)
    TLinears2.addLegend()
    TppoTmmLinear = TLinears2.plot('ko-', name="|T++/T--|")
    TpmoTmpLinear = TLinears2.plot('ro-', name="|T+-/T+-|")

    TppoTmmALinear = TLinears2.plot('ko--', name="ph(T++/T--)")
    double2.p2.addItem(TppoTmmALinear)
    TpmoTmpALinear = TLinears2.plot('ro--', name="ph(T+-/T-+)")
    double2.p2.addItem(TpmoTmpALinear)

    tabWid.addTab(TLinears2, "T Matrices (Ratios)")

    cboxGroup.buttonToggled.connect(updateJ)

    layout.addWidget(tabWid)
    mainwid.setLayout(layout)

    return mainwid
