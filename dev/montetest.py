import monteCarloObject as mco

monte = mco.monteCarlo(alphas='./12-05_alphasgammas_alpha.txt',
                       gammas='./12-05_alphasgammas_gamma.txt',
                       folder_name='theta9001',
                       observedSidebands=(8, 30))
