import numpy as np

np.set_printoptions(linewidth=500)


class Berry(object):
    """
    w = [Theta, k, (k,n)]
    v = [Theta, k, (u1...u4,x,y,z), (uij)]
    du = [k, n, m, i]
    A = [Theta, k, n, m, i]
    dA = [k, n, m, i, j] = djAi
    O = [Theta, k, n, m, i, j]
    Phase = [Theta, n]
    """
    def __init__(self, g1, g2, g3, steps, angSteps, below=True):
        """
       The Berry Class is used for calculating the Berry physics from the
       defined parameters of the Luttinger Hamiltonian.

       Init will initialize the class and carry through values used throughout
       the processing

       11/19/2020 I'm not sure how much of this was ever used or if this works
       but as I'm getting things ready for refactoring, I figured it would be
       worth carrying this along. Most Berry phase/curvature calculations are
       usually handled by Mathematica, but this stuff is here if someone wants.

       :param self - the object to be used to calculate the Berry properties
       :param g1 - the Gamma1 Luttinger parameter
       :param g2 - the Gamma2 Luttinger parameter
       :param g3 - the Gamma3 Luttinger parameter
        """
        # Qile's way of calculating the Kane gamma factors
        P = 10.493      # eV*A, Conduction band fit of y=p*x^2
        a_bhor = 0.53   # A, Bohr Radius
        ryd = 13.6      # eV, Rydberg
        P_eff = (P/a_bhor)**2/ryd
        E_g = 1.506     # eV, Gap Energy
        # Kane Parameters
        g18 = g1 - P_eff/(3*E_g)
        g28 = g2 - P_eff/(6*E_g)
        g38 = g3 - P_eff/(6*E_g)
        self.g1 = g18
        self.g2 = g28
        self.g3 = g38

        self.st = steps
        self.ang = angSteps
        self.below = below

        self.w = np.zeros((self.ang, self.st, 5))
        self.v = np.zeros((self.ang, self.st, 7, 4))
        if below:
            self.A = np.zeros((self.ang, self.st, 4, 4, 2))
            self.O = np.zeros((self.ang, self.st, 4, 4, 2, 2))
        else:
            self.A = np.zeros((self.ang, self.st, 4, 4, 3))
            self.O = np.zeros((self.ang, self.st, 4, 4, 3, 3))

    def Luttinger(self, theta, BZfrac):
        '''
        Calculates the Luttinger Hamiltonian based on the input parameters

        :theta: Sample orientation with respect to the [010] Axis
        :BZfrac: The fraction of the Brillouin Zone calculated
        '''
        th = theta*np.pi/180  # radians
        BZfrac
        angIdx = np.int(theta*self.ang/360)
        # Spin Matrices
        Sx = np.array([[0, np.sqrt(3)/2, 0, 0],
                       [np.sqrt(3)/2, 0, 1, 0],
                       [0, 1, 0, np.sqrt(3)/2],
                       [0, 0, np.sqrt(3)/2, 0]])
        Sy = np.array([[0, np.sqrt(3)/(2*1j), 0, 0],
                       [-np.sqrt(3)/(2*1j), 0, (1/1j), 0],
                       [0, -(1/1j), 0, np.sqrt(3)/(2*1j)],
                       [0, 0, -np.sqrt(3)/(2*1j), 0]])
        Sz = np.array([[3/2, 0, 0, 0],
                       [0, 1/2, 0, 0],
                       [0, 0, -1/2, 0],
                       [0, 0, 0, -3/2]])

        # Pauli Matrices
        p0 = np.array([[1, 0], [0, 1]])
        px = np.array([[0, 1], [1, 0]])
        py = np.array([[0, -1j], [1j, 0]])
        # pz = np.array([[1, 0], [0, -1]])

        # Fraction of Brilioun Zone Traversed
        kmax = BZfrac*2*np.pi/(5.6325)
        hbar = 1  # 1.054572 * 10**(-34) #m^2 kg/s
        # hbarc = 0.197326 * 10**(4)  # Angstrom eV
        eMass = 9.109383  # kg
        NIRWavelength = 8230  # Angstrom

        h = np.zeros((self.st, 4, 4))
        i = 0

        if self.below:
            for k in np.arange(0, kmax, kmax/self.st):
                kx = k*np.cos(th)
                ky = k*np.sin(th)
                h[i, 0:2, 0:2] = np.array(-hbar**2/(2*eMass)*(self.g1*(
                                          kx**2+ky**2)*p0-2*self.g2*(
                                          np.sqrt(3)*(kx**2-ky**2)/2)*px +
                                          2*np.sqrt(3)*self.g3*kx*ky*py +
                                          self.g2*(kx**2+ky**2)))
                h[i, 2:4, 2:4] = np.array(-hbar**2/(2*eMass)*(self.g1*(
                                          kx**2+ky**2)*p0 -
                                          self.g2*(np.sqrt(3)*(kx**2-ky**2))*px
                                          - 2*np.sqrt(3)*self.g3*kx*ky*py -
                                          self.g2*(kx**2+ky**2)))
                self.w[angIdx, i, 1:5], self.v[angIdx, i, 0:4, :] = (
                                                    np.linalg.eig(h[i, :, :]))
                self.w[angIdx, i, 1:5] = np.absolute(self.w[angIdx, i, 1:5])
                self.w[angIdx, i, 1:5] = np.sort(self.w[angIdx, i, 1:5])
                self.w[angIdx, i, 0] = k
                self.v[angIdx, i, 4, :] = kx
                self.v[angIdx, i, 5, :] = ky
                self.v[angIdx, i, 6, :] = 0
                i = i+1
        else:
            for k in np.arange(0, kmax, kmax/self.st):
                kx = k*np.cos(th)
                ky = k*np.sin(th)
                kz = (1/(2*np.pi*NIRWavelength))-(1/(2*np.pi*8225))
                h[i, :, :] = (np.array(-hbar**2/(2*eMass)*((
                              self.g1+5/2*self.g2)*k**2 -
                              2*self.g3*(kx*Sx+ky*Sy+kz*Sz)**2 +
                              2*(self.g3-self.g2) *
                              (kx**2*Sx**2 + ky**2*Sy**2+kz**2*Sz**2))))
                self.w[angIdx, i, 1:5], self.v[angIdx, i, 0:4, :] = (
                                                    np.linalg.eig(h[i, :, :]))
                self.w[angIdx, i, 1:5] = np.absolute(self.w[angIdx, i, 1:5])
                self.w[angIdx, i, 1:5] = np.sort(self.w[angIdx, i, 1:5])
                self.w[angIdx, i, 0] = k
                self.v[angIdx, i, 4, :] = kx
                self.v[angIdx, i, 5, :] = ky
                self.v[angIdx, i, 6, :] = kz
                i = i+1

    def LuttingerUbasis(self, theta, BZfrac):
        '''
        Calculates the Luttinger Hamiltonian based on the input parameters
            in the Bloch basis
        :theta: Sample orientation with respect to the [010] Axis
        :BZfrac: The fraction of the Brillouin Zone calculated
        '''
        th = theta*np.pi/180  # radians
        self.BZf = BZfrac
        angIdx = np.int(theta*self.ang/360)
        # Fraction of Brilioun Zone Traversed
        kmax = self.BZf * 2*np.pi/(5.6325*10**(-10))
        hbar = 1.054572 * 10**(-34)  # m^2 kg/s
        # hbarc = 0.197326 * 10**(4)  # Angstrom eV
        eMass = 9.109383 * 10**(-31)  # kg
        NIRWavelength = 8230  # Angstrom

        h = np.zeros((self.st, 4, 4))
        i = 0

        if self.below:
            for k in np.arange(0, kmax, kmax/self.st):
                kx = k*np.cos(th)
                ky = k*np.sin(th)
                h[i, 0, 0] = -hbar**2/(2*eMass)*((self.g2 + self.g1)*(
                            kx**2 + ky**2))
                h[i, 0, 1] = -hbar**2/(2*eMass)*(-np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)-2j*self.g3*kx*ky))
                h[i, 0, 2] = 0
                h[i, 0, 3] = 0
                h[i, 1, 0] = -hbar**2/(2*eMass)*(-np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)+2j*self.g3*kx*ky))
                h[i, 1, 1] = -hbar**2/(2*eMass)*((self.g1 - self.g2)*(
                            kx**2 + ky**2))
                h[i, 1, 2] = 0
                h[i, 1, 3] = 0
                h[i, 2, 0] = 0
                h[i, 2, 1] = 0
                h[i, 2, 2] = -hbar**2/(2*eMass)*((self.g1 - self.g2)*(
                            kx**2 + ky**2))
                h[i, 2, 3] = -hbar**2/(2*eMass)*(-np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)+2j*self.g3*kx*ky))
                h[i, 3, 0] = 0
                h[i, 3, 1] = 0
                h[i, 3, 2] = -hbar**2/(2*eMass)*(-np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)-2j*self.g3*kx*ky))
                h[i, 3, 3] = -hbar**2/(2*eMass)*((self.g1 + self.g2)*(
                            kx**2 + ky**2))
                # print(h)
                self.w[angIdx, i, 1:5], self.v[angIdx, i, 0:4, :] = (
                    np.linalg.eig(h[i, :, :]))
                self.w[angIdx, i, 0] = k*(5.6325*10**(-10))/(2*np.pi)
                self.v[angIdx, i, 4, :] = kx
                self.v[angIdx, i, 5, :] = ky
                self.v[angIdx, i, 6, :] = 0
                i = i+1
        else:
            for k in np.arange(0, kmax, kmax/self.st):
                kx = k*np.cos(th)
                ky = k*np.sin(th)
                kz = (1/(2*np.pi*NIRWavelength))-(1/(2*np.pi*8225))
                h[i, 0, 0] = hbar**2/(2*eMass)*(-(self.g2 + self.g1)*(
                            kx**2 + ky**2) - (self.g1 - 2*self.g2)*kz**2)
                h[i, 0, 1] = hbar**2/(2*eMass)*(2*np.sqrt(3)*self.g3*kz*(
                            kx - 1j*ky))
                h[i, 0, 2] = hbar**2/(2*eMass)*(np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)-2j*self.g3*kx*ky))
                h[i, 0, 3] = 0
                h[i, 1, 0] = hbar**2/(2*eMass)*(2*np.sqrt(3)*self.g3*kz*(
                            kx + 1j*ky))
                h[i, 1, 1] = hbar**2/(2*eMass)*(-(self.g1 - self.g2)*(
                            kx**2 + ky**2) - (self.g1 + 2*self.g2)*kz**2)
                h[i, 1, 2] = 0
                h[i, 1, 3] = hbar**2/(2*eMass)*(np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)-2j*self.g3*kx*ky))
                h[i, 2, 0] = hbar**2/(2*eMass)*(np.sqrt(3)*(self.g2*(
                            kx**2-ky**2)+2j*self.g3*kx*ky))
                h[i, 2, 1] = 0
                h[i, 2, 2] = hbar**2/(2*eMass)*(-(self.g1-self.g2)*(
                            kx**2 + ky**2) - (self.g1 + 2*self.g2)*kz**2)
                h[i, 2, 3] = -hbar**2/(2*eMass)*(2*np.sqrt(3)*self.g3*kz*(
                            kx - 1j*ky))
                h[i, 3, 0] = 0
                h[i, 3, 1] = hbar**2/(2*eMass)*(np.sqrt(3)*(self.g2*(
                            kx**2 - ky**2) + 2j*self.g3*kx*ky))
                h[i, 3, 2] = -hbar**2/(2*eMass)*(2*np.sqrt(3)*self.g3*kz*(
                            kx + 1j*ky))
                h[i, 3, 3] = hbar**2/(2*eMass)*(-(self.g1 + self.g2)*(
                            kx**2 + ky**2) - (self.g1 - 2*self.g2)*kz**2)
                self.w[angIdx, i, 1:5], self.v[angIdx, i, 0:4, :] = (
                    np.linalg.eig(h[i, :, :]))
                self.w[angIdx, i, 1:5] = np.absolute(self.w[angIdx, i, 1:5])
                self.w[angIdx, i, 1:5] = np.sort(self.w[angIdx, i, 1:5])
                self.w[angIdx, i, 0] = k
                self.v[angIdx, i, 4, :] = kx
                self.v[angIdx, i, 5, :] = ky
                self.v[angIdx, i, 6, :] = kz
                i = i+1

        return self.v, self.w

    def NABerryConnection(self, theta):
        """
        Takes in the four conduction band eigenfuntions of the Luttinger
        Hamiltonian and calculates the non-Abelian Berry Connection as a
        function of K.

        :param v = (steps,7,4) matrix, (4X4) eigenfunctions and (3X4) kx, ky kz
        """
        th = np.int(theta*self.ang/360)
        if self.below:
            # Initialize the Berry Connection Matrix
            # We are technically meshing in Radial Coordinates, so we are going
            # to have to pull some Jacobian
            # nonsense to get everything in the proper working order.
            du = np.zeros((self.st-1, 4, 4, 2))

            if th == 0:
                for n in range(0, 4, 1):
                    du[0, n, :, 0] = (self.v[59, 0, n, :]-self.v[1, 0, n, :])/(
                        2*np.pi/self.ang)*self.w[th, 0, 0]*np.sin(
                        np.pi*theta/180)
                    du[0, n, :, 1] = (self.v[59, 0, n, :]-self.v[1, 0, n, :])/(
                        2*np.pi/self.ang)*self.w[th, 0, 0]*np.cos(
                        np.pi*theta/180)
            elif th == 59:
                for n in range(0, 4, 1):
                    du[0, n, :, 0] = (self.v[0, 0, n, :]-self.v[58, 0, n, :])/(
                        2*np.pi/self.ang)*self.w[th, 0, 0]*np.sin(
                        np.pi*theta/180)
                    du[0, n, :, 1] = (self.v[0, 0, n, :]-self.v[58, 0, n, :])/(
                        2*np.pi/self.ang)*self.w[th, 0, 0]*np.cos(
                        np.pi*theta/180)
            else:
                for n in range(0, 4, 1):
                    du[0, n, :, 0] = (self.v[th+1, 0, n, :] -
                                      self.v[th-1, 0, n, :])/(
                                      2*np.pi/self.ang)*self.w[th, 0, 0]*(
                                      np.sin(np.pi*theta/180))
                    du[0, n, :, 1] = (self.v[th+1, 0, n, :] -
                                      self.v[th-1, 0, n, :])/(
                                      2*np.pi/self.ang)*self.w[th, 0, 0]*(
                                      np.cos(np.pi*theta/180))

            if th == 0:
                for k in range(1, self.st-1, 1):
                    for n in range(0, 4, 1):
                        # 0 = x, 1 = y;
                        du[k, n, :, 0] = ((
                            self.v[th, k+1, n, :]-self.v[th, k-1, n, :])/(
                            self.w[th, k+1, 0]-self.w[th, k-1, 0])*np.cos(
                            np.pi*theta/180)-(
                            self.v[59, k, n, :]-self.v[1, k, n, :])/(
                            2*np.pi/self.ang)*self.w[th, k, 0]*np.sin(
                            np.pi*theta/180))
                        du[k, n, :, 1] = ((
                            self.v[th, k+1, n, :]-self.v[th, k-1, n, :])/(
                            self.w[th, k+1, 0]-self.w[th, k-1, 0])*np.sin(
                            np.pi*theta/180) + (
                            self.v[59, k, n, :]-self.v[1, k, n, :])/(
                            2*np.pi/self.ang)*self.w[th, k, 0]*np.cos(
                            np.pi*theta/180))
                    # Finding the derivative of the Bloch functions as each
                    # point in K-space.
            elif th == 59:
                for k in range(1, self.st-1, 1):
                    for n in range(0, 4, 1):
                        # 0 = x, 1 = y;
                        du[k, n, :, 0] = (
                            self.v[th, k+1, n, :]-self.v[th, k-1, n, :])/(
                            self.w[th, k+1, 0]-self.w[th, k-1, 0])*np.cos(
                            np.pi*theta/180) - (
                            self.v[1, k, n, :]-self.v[59, k, n, :])/(
                            2*np.pi/self.ang)*self.w[th, k, 0]*np.sin(
                            np.pi*theta/180)
                        du[k, n, :, 1] = (
                            self.v[th, k+1, n, :]-self.v[th, k-1, n, :])/(
                            self.w[th, k+1, 0]-self.w[th, k-1, 0])*np.sin(
                            np.pi*theta/180) + (
                            self.v[1, k, n, :]-self.v[59, k, n, :])/(
                            2*np.pi/self.ang)*self.w[th, k, 0]*np.cos(
                            np.pi*theta/180)
                    # Finding the derivative of the Bloch functions as each
                    # point in K-space.
            else:
                for k in range(1, self.st-1, 1):
                    for n in range(0, 4, 1):
                        # 0 = x, 1 = y;
                        du[k, n, :, 0] = (
                            self.v[th, k+1, n, :]-self.v[th, k-1, n, :])/(
                            self.w[th, k+1, 0]-self.w[th, k-1, 0])*np.cos(
                            np.pi*theta/180) - (
                            self.v[th+1, k, n, :]-self.v[th-1, k, n, :])/(
                            2*np.pi/self.ang)*self.w[th, k, 0]*np.sin(
                            np.pi*theta/180)
                        du[k, n, :, 1] = (
                            self.v[th, k+1, n, :]-self.v[th, k-1, n, :])/(
                            self.w[th, k+1, 0]-self.w[th, k-1, 0])*np.sin(
                            np.pi*theta/180) + (
                            self.v[th+1, k, n, :]-self.v[th-1, k, n, :])/(
                            2*np.pi/self.ang)*self.w[th, k, 0]*np.cos(
                            np.pi*theta/180)
                    # Finding the derivative of the Bloch functions as each
                    # point in K-space.

            for k in range(0, self.st-1, 1):
                for n in range(0, 4, 1):
                    for m in range(0, 4, 1):
                        for i in range(0, 2, 1):
                            self.A[th, k, n, m, i] = (
                                self.v[th, k, n, 0]*du[k, m, 0, i] +
                                self.v[th, k, n, 1]*du[k, m, 1, i] +
                                self.v[th, k, n, 2]*du[k, m, 2, i] +
                                self.v[th, k, n, 3]*du[k, m, 3, i])
            return self.A
        else:
            # # Initialize the Berry Connection Matrix with 3 Cartesian
            # #     Coordinates
            # du = np.zeros((self.st,4,3))
            # for k in range(1,self.st-1,1):
            #     for n in range(0,4,1):
            #             # 0 = x, 1 = y, 2 = z;
            #         du[k,n,:,0] = (self.v[th,k+1,n,:]-self.v[th,k-1,n,:])/(
            #           self.u[th,k+1,0]-self.u[th,k-1,0])*np.cos(
            #           np.pi*theta/180) - ...
            #             (self.v[th+1,k,n,:]-self.v[th-1,k,n,:])/(
            #              2*np.pi/self.ang)*self.u[th,k,0]*np.sin(
            #              np.pi*theta/180)
            #         du[k,n,:,1] = (self.v[th,k+1,n,:]-self.v[th,k-1,n,:])/(
            #           self.u[th,k+1,0]-self.u[th,k-1,0])*np.sin(
            #           np.pi*theta/180) + ...
            #             (self.v[th+1,k,n,:]-self.v[th-1,k,n,:])/(
            #             2*np.pi/self.ang)*self.u[th,k,0]*np.cos(
            #             np.pi*theta/180)
            #         du[k,n,:,2] = (self.v[th,k+1,n,:])

            for k in range(0, self.st-1, 1):
                for n in range(0, 4, 1):
                    for m in range(0, 4, 1):
                        for i in range(0, 3, 1):
                            self.A[th, k, n, m, i] = (
                                self.v[th, k, n, 0]*du[k, m, 0, i] +
                                self.v[th, k, n, 1]*du[k, m, 1, i] +
                                self.v[th, k, n, 2]*du[k, m, 2, i] +
                                self.v[th, k, n, 3]*du[k, m, 3, i])

    def NABerryCurvature(self, theta):
        """
        Calculate the Berry Curvature using the calculated Berry Connection
        Array.
        theta - the angle where the Hamiltonian is calculated.
        """
        th = np.int(theta*self.ang/360)
        # Below gap (Kz = 0)
        if self.below:
            # Initialize the Berry Curvature Matrix and Array for Derivative of
            #   A
            dA = np.zeros((self.st, 4, 4, 2, 2))

            # Calculate the derivative of the Berry Connection
            for k in range(1, self.st-1, 1):
                for m in range(0, 4, 1):
                    for n in range(0, 4, 1):
                        for i in range(0, 2, 1):          # Ai
                            for j in range(0, 2, 1):      # dj
                                dA[k-1, m, n, i, j] = (
                                    self.A[th, k+1, m, n, i] -
                                    self.A[th, k-1, m, n, i])/(
                                    self.v[th, k+1, 4+j, 0] -
                                    self.v[th, k-1, 4+j, 0])

            # Calculate the Berry Curvature
            for k in range(0, self.st-1, 1):
                for m in range(0, 4, 1):
                    for n in range(0, 4, 1):
                        for i in range(0, 2, 1):
                            for j in range(0, 2, 1):
                                self.O[th, k, m, n, i, j] = (
                                    dA[k, m, n, j, i] - dA[k, m, n, i, j])
            return self.O
        else:   # Above Gap (kz neq 0)
            dA = np.zeros((self.st, 4, 4, 3))
            for k in range(1, self.st, 1):
                for m in range(0, 4, 1):
                    for n in range(0, 4, 1):
                        for i in range(0, 3, 1):          # Ai
                            for j in range(0, 3, 1):      # dj
                                dA[k-1, m, n, i, j] = (
                                    self.A[th, k+1, m, n, i] -
                                    self.A[th, k-1, m, n, i])/(
                                    self.v[th, k+1, 4+j, 0] -
                                    self.v[th, k-1, 4+j, 0])

            for k in range(0, self.st-1, 1):
                for m in range(0, 4, 1):
                    for n in range(0, 4, 1):
                        for i in range(0, 2, 1):
                            for j in range(0, 2, 1):
                                self.O[th, k, m, n, i, j] = (
                                    dA[k, m, n, j, i] - dA[k, m, n, i, j])

    def BerryMesh(self, BZf):
        """
        Calculates the Berry Values at various theta to create a radial mesh.
        """
        angInc = np.int(360/self.ang)
        # First create a full mesh of Luttinger Params
        for theta in range(0, 360, angInc):
            self.LuttingerUbasis(theta, BZf)

        # Use the Luttinger Values to create Berry Values
        for theta in range(0, 360, angInc):
            self.NABerryConnection(theta)
            self.NABerryCurvature(theta)
        return self.A, self.O

    def NABerryPhase(self, bands):
        """
        Calculate the Berry Phase from the Berry Mesh Calculation
        """

        Phase = np.zeros((self.st, bands + 1))
        for k in range(0, self.st, 1):
            dk = k*self.BZf/self.st*2*np.pi/(5.6325)
            for n in range(0, bands, 1):
                for m in range(0, bands, 1):
                    for t in range(0, self.ang, 1):
                        theta = t*self.ang*np.pi/180
                        Phase[k, n+1] = (
                            Phase[k, n+1] + self.A[t, k, n, m, 1] *
                            dk*self.ang*np.pi/180*np.cos(theta) -
                            self.A[t, k, n, m, 0]*dk*self.ang*np.pi/180 *
                            np.sin(theta))
            Phase[k, 0] = self.w[0, k, 0]
        return Phase
