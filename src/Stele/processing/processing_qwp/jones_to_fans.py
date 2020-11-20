import numpy as np
from Stele.processing.processing_jones.jones_vector import JonesVector as JV


def jonesToFans(sbs, J, wantNIR=np.arange(-90, 90, 5)):
    """
    Take a Jones matrix as an Nx2x2. Returns alpha and gamma matrices similar
        to those produced by FanCompiler.build(), which can be passed directly
        to a the FanDiagram constructor
    :param J:
    :return:
    """
    vec = JV(alpha=wantNIR, gamma=0)
    vec.apply_transformation(J)

    alphas = np.column_stack((sbs, vec.alpha))
    alphas = np.row_stack(([-1] + wantNIR.tolist(), alphas))

    gammas = alphas.copy()
    gammas[1:, 1:] = vec.gamma

    return alphas, gammas
