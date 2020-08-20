class SbStateGetter(object):
    """
    sister function to FanCompiler. Useful for taking the arrays out of a FanCompiler
    allowing indexing based on sb number and nir alpha angles.

    Example creation:
    fc = FanCompiler(wantSBs)
    // initialize fc

    getter = SbStateGetter(
        fc.arrA[:, 1:],
        fc.arrG[:, 1:],
        fc.want,
        fc.nirAlphas
    )
    """
    def __init__(self, alphas, gammas, sbNum, niralphas):
        self.alphas = alphas
        self.gammas = gammas
        self.sbNum = sbNum
        self.niralphas = niralphas

        self.invS = {kk: ii for ii, kk in enumerate(sbNum)}

        self.invA = {kk: ii for ii, kk in enumerate(niralphas)}

    def getSBState(self, sb, nirA):
        alpha = self.alphas[self.invS[sb], self.invA[nirA]]
        gamma = self.gammas[self.invS[sb], self.invA[nirA]]

        return alpha, gamma

    def __call__(self, sb, nirA):
        return self.getSBState(sb, nirA)

    def getStateDict(self, sb, nirA):
        a, g = self.getSBState(sb, nirA)
        return {"alpha": a, "gamma": g}
