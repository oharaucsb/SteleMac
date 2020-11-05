import numpy as np
import hsganalysis.newhsganalysis as hsg
import hsganalysis.QWPProcessing as qwp
from PyQt5 import QtWidgets

"""

This first bit does things a little more explicitly to better understand
how all of the data processing works.

"""


# Make a list of all the folders for this fan diagram
datasets = hsg.natural_glob(r"FanData", "*")
# Which sidebands did you see? Data gets cropped so everything has these
#   sidebands inclduded
observedSidebands = np.arange(10, 32, 2)

# You need the crystal angle to get the T matrix
crystalAngle = -123+45

# Name to save things with
saveFileName = "820nm_c123"

# FanCompiler is a class which adds together the different \alpha and \gamma of
# for different \alpha_NIR, and then turns it into a
fanData = qwp.expFanCompiler.FanCompiler(observedSidebands)

for data in datasets:
    # Do the standard data processing
    # You should know from previous experiments how to call these functions
    laserParams, rawData = hsg.hsg_combine_qwp_sweep(data)
    _, fitDict = hsg.proc_n_fit_qwp_data(
        rawData, vertAnaDir=False, laserParams=laserParams,
        wantedSbs=observedSidebands
        )
    # Add the new alpha and gamma sets to the fan class
    fanData.addSet(fitDict)

# Building the fan compiler causes it to make  2D np arrays with the alpha and
# gamma angles where each column is a different alpha_NIR excitation
#
# The arguments for the savename are weird because I don't usually save
# this data
fanData.buildAndSave(saveFileName + "_{}.txt")

# load the data that got saved.
# Alternatively, you can use
#    alphaData, gammaData, _ = fanData.build()
# if you don't want to bother saving the text file
alphaData = np.genfromtxt(
    saveFileName + "_{}.txt".format("alpha"), delimiter=','
    )
gammaData = np.genfromtxt(
    saveFileName + "_{}.txt".format("gamma"), delimiter=','
    )

# Now we need to calculate the Jones matrices.
# This is a
J = qwp.extractMatrices.findJ(alphaData, gammaData)

# A lot of stuff can be done with this Jones matrix now....

# Get the T matrix:
T = qwp.extractMatrices.makeT(J, crystalAngle)

# Either file can be saved:
# (It's called saveT, but it works for any complex 2x2 matrix, including J).
qwp.extractMatrices.saveT(
    J, observedSidebands, "{}_JMatrix.txt".format(saveFileName)
    )
qwp.extractMatrices.saveT(
    T, observedSidebands, "{}_TMatrix.txt".format(saveFileName)
    )

app = QtWidgets.QApplication([])

# Interpolate all alpha and gamma from J
resampledAlpha, resampledGamma = qwp.expFanCompiler.jonesToFans(
    observedSidebands, J
    )

# Make Fan diagram
f = qwp.fanDiagram.FanDiagram(resampledAlpha, resampledGamma)
f.setTitle(title=saveFileName, adjustBounds=False)
f.show()
app.exec_()

# You can save the fan with
f.export(saveFileName+"_fanDiagram.png")


"""

And since a lot of this code is pretty automatable, things have been condensed
down in a lot of functions.

Loading the raw experimental data is done automatically with

fanData = qwp.expFanCompiler.FanCompiler.fromDataFolder(
    "Fan Data", observedSidebands
    )

Fan diagrams can be made directly from T matrices,
    f = qwp.fanDiagram.FanDiagram.fromTMatrix()

"""
