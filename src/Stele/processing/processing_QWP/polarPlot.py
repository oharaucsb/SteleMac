from __future__ import division
import numpy as np
import Stele.ipg as pg
# import pyqtgraph as pg

def makePolarPlot():
    plot = pg.figure()
    plot.setAspectLocked()
    # Add polar grid lines
    plot.addLine(x=0, pen=0.2)
    plot.addLine(y=0, pen=0.2)
    for r in range(4, 38, 2):
        circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        circle.setPen(pg.pg.mkPen(0.2, width=1+1*(r==18)))
        plot.addItem(circle)
    return plot


def polarPlot(*args, **kwargs):

    plt = pg.gcf()

    crv = plt.plot(*args, **kwargs)

    old = crv.setData
    def newSetData(r, theta, *args, **kwargs):

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return old(x, y, *args, **kwargs)

    crv.setData = newSetData

    return crv
def polarPlotBad(*args, **kwargs):

    plt = pg.gcf()

    crv = plt.plot()

    old = crv.setData
    def newSetData(self, *args, **kwargs):
        old(self, *args, **kwargs)
        x = self.xData * np.cos(self.yData)
        y = self.xData * np.sin(self.yData)

    crv.setData = newSetData

    crv.setData(*args, **kwargs)

    return crv
