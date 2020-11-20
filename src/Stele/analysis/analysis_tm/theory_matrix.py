import os
import time
import numpy as np
import scipy.special as spl
import scipy.integrate as intgt
from Stele.processing.processing_qwp.extract_matrices import makeT, saveT

np.set_printoptions(linewidth=500)


class TheoryMatrix(object):
    def __init__(self, ThzField, Thzomega, nir_wl, dephase, detune, temp=60):
        '''
        This class is designed to handle everything for creating theory
        matrices and comparing them to experiement.

        Init defines some constants that are used throughout the calculation
        and puts somethings in proper units.

        Parameters:
        :ThzField: Give in kV/cm.
        :Thzomega: Give in Ghz.
        :nir_wl: Give in nanometers.
        :dephase: Dephasing, give in meV.
            Should roughly be the width of absorption peaks
        :detune: Detuning, give in meV.
            Difference between NIR excitation and band gap
        :temp: Temperature, give in K
        '''

        self.F = ThzField * 10**5
        self.Thz_w = Thzomega * 10**9 * 2 * np.pi
        self.nir_wl = nir_wl * 10**(-9)
        self.dephase = dephase * 1.602*10**(-22)
        self.detune = detune*1.602*10**(-22)
        self.n_ref = 0
        self.iterations = 0
        self.max_iter = 0
        self.hbar = 1.055*10**(-34)  # hbar in Js
        self.temp = temp
        self.kb = 8.617*10**(-5)  # Boltzmann constant in eV/K
        self.temp_ev = self.temp*self.kb

    def mu_generator(self, gamma1, gamma2, phi, beta):
        '''
        Given gamma1 and gamma2 produces mu+- according to

        mu+- = electron mass/(mc^-1+gamma1 -+ 2*gamma2)

        Note that this formula is only accurate for THz and NIR
        polarized along [010]. The general form requires gamma3 as well

        Parameters:
        :gamma1: Gamma1 parameter in the luttinger hamiltonian.
            Textbook value of 6.85
        :gamma2: Gamma2 parameter in the luttinger hamiltonian.
            Textbook value of 2.1
        :phi: [100] to THz orientation, passed from the data array
        :beta: experimentally measured g3/g2 ratio

        Returns: mu_p, mu_m effective mass of of mu plus/minus
        '''
        theta = phi + np.pi/4
        emass = 9.109*10**(-31)  # bare electron mass in kg
        m_cond = 0.0665  # Effective mass of conduction band

        mu_p = emass/(
            1/m_cond + gamma1 - gamma2*np.sqrt(
                3*np.sin(2*theta)**2+1+3*np.cos(2*theta)**2*beta**2))
        # Calculates mu_plus
        mu_m = emass/(
            1/m_cond + gamma1 + gamma2*np.sqrt(
                3*np.sin(2*theta)**2+1+3*np.cos(2*theta)**2*beta**2))
        # Calculates mu_minus

        return mu_p, mu_m

    def alpha_value(self, x):
        '''
        alpha parameter given by Qile's notes on two band model for a given x

        Parameters:
        :x: the argument of the calculation. Give in radians

        Returns:
        :alpha_val: the alpha parameter given in Qile's notes
        '''

        alpha_val = np.cos(x/2) - np.sin(x/2)/(x/2)
        # This does the calculation. Pretty straightforward

        return alpha_val

    def gamma_value(self, x):
        '''
        gamma parameter given by Qile's notes on two band model

        Parameters:
        :x: Argument of the calculation. Give in radians

        Returns:
        :gamma_val: the gamma parameter given in Qile's notes
        '''

        gamma_val = np.sin(x/2)/(x/2)
        # does the calculation

        return gamma_val

    def Up(self, mu):
        '''
        Calculates the ponderemotive energy
        Ponderemotive energy given by

        U = e^2*F_THz^2/(4*mu*w_THz^2)

        Parameters:
        :F: Thz field. Give in V/m
        :mu: effective mass. Give in kg
        :w: omega,  the THz freqeuncy. Give in angular frequency.

        Returns:
        :u: The ponderemotive energy
        '''
        F = self.F
        w = self.Thz_w

        echarge = 1.602*10**(-19)  # electron charge in Coulombs

        u = echarge**(2)*F**(2)/(4*mu*w**2)
        # calculates the ponderemotive energy

        return u

    def phonon_dephase(self, n):
        '''
        Step function that will compare the energy gained by the sideband to
        the energy of the phonon (36.6meV). If the energy is less than the
        phonon, return zero.
        If it's more return the scattering rate as determined by
        Yu and Cordana Eq 5.51

        This really should be treated as a full integral, but whatever
        '''

        thz_omega = self.Thz_w
        hbar = self.hbar
        thz_ev = n*hbar*thz_omega/(1.602*10**-19)  # converts to eV
        phonon_ev = 36.6*10**(-3)  # phonon energy in Vv

        # emass = 9.109*10**(-31)  # bare electron mass in kg
        # m_cond = 0.0665  # Effective mass of conduction band
        # m_eff = emass*m_cond

        phonon_n = 1/(np.exp(phonon_ev/self.temp_ev)-1)

        if thz_ev < phonon_ev:
            # print('No phonon for order',n)
            return 0

        else:
            W0 = 7.7*10**12  # characteristic rate

            rate_frac = phonon_n*np.sqrt((thz_ev+phonon_ev)/thz_ev)+(
                phonon_n+1)*np.sqrt((thz_ev-phonon_ev)/thz_ev)+(
                phonon_ev/thz_ev)*(-phonon_n*np.arcsinh(np.sqrt(
                    phonon_ev/thz_ev))+(phonon_n+1)*np.arcsinh(np.sqrt(
                        (thz_ev-phonon_ev)/thz_ev)))
            # Got this from Yu and Cordana's book

            fullW = W0*rate_frac

            return fullW

    def integrand(self, x, mu, n):
        '''
        Calculate the integrand to integrate A_n+- in two_band_model pdf eqn 13
        Given in the new doc pdf from Qile as I_d^(2n)

        Parameters:
        :x: Argument of integrand equal to omega*t. This is the variable
            integrated over.
        :dephase: dephasing rate. Should be a few meV, ~the width of the
            exciton absorption peak (according to Qile). Should be float
        :w: Frequency of THz in radians.
        :F: Thz field in V/m
        :mu: reduced mass give in kg
        :n: Order of the sideband

        Returns:
        :result: The value of the integrand for a given x value
        '''
        hbar = self.hbar
        # F = self.F
        w = self.Thz_w
        dephase = self.dephase
        detune = self.detune
        pn_dephase = self.phonon_dephase(n)

        exp_arg = (-dephase*x/(hbar*w)-pn_dephase*x/w + 1j*x*self.Up(mu)/(
            hbar*w)*(self.gamma_value(x)**2-1)+1j*n*x/2-1j*detune*x/(hbar*w))
        # Argument of the exponential part of the integrand

        bessel_arg = x*self.Up(mu)*self.alpha_value(x)*self.gamma_value(x)/(
            hbar*w)
        # Argument of the bessel function

        bessel = spl.jv(n/2, bessel_arg)
        # calculates the J_n(bessel_arg) bessel function

        result = np.exp(exp_arg)*bessel/x
        # This is the integrand for a given x

        return result

    def Qintegrand(self, x, mu, n):
        '''
        Calculate the integrand in the expression for Q, with the
        simplifixation that the canonical momentum is zero upon exciton pair
        creation.

        Parameters:
        :x: integration variable of dimensionless units. Equal to omega*tau
        :dephase: dephasing rate of the electron hole pair as it is accelerated
        by the THz field
        :w: Frequency of THz is radiams
        :F: THz field in V/m
        :mu: the effective reduced mass of the electron-hole pair
        :n: Order of the sideband
        '''
        hbar = self.hbar
        # F = self.F
        w = self.Thz_w
        dephase = self.dephase
        # detune = self.detune
        pn_detune = self.phonon_dephase(n)

        exp_arg = (-dephase*x - pn_detune*x/w + 1j*x*(self.Up(mu)/(hbar*w)+n))

        bessel_arg = self.Up(mu)*np.sin(x)/w

        bessel = spl.jv(n/2, bessel_arg)

        result = np.exp(exp_arg)*bessel/x

        return result

    def scale_J_n_T(
     self, Jraw, Jxx, observedSidebands, crystalAngle,
     saveFileName, index, save_results=True, scale_to_i=True):
        '''
        This function takes the raw J from fan_n_Tmat or findJ and scales it
        with Jxx found from scaling sideband strengths with the laser line/PMT

        In regular processing we actually find all the matrices normalized to
        Jxx

        Now can scale to a given sideband order.
        This is to allow comparision between the measured sideband powers,
        normalized by the PMT, to the evalueated Path Integral from the two
        band model. By normalizing the measured values and integrals to a given
        sideband index, we can remove the physical constants from the
        evaluation.

        :param Jraw: set of matrices from findJ
        :param Jxx: sb_results from PMT and CCD data
        :param observedSidebands: np array of observed sidebands. Data will be
            cropped such that these sidebands are included in everything.
        :param crystalAngle: (Float) Angle of the sample from the 010 crystal
            face
        :saveFileName: Str of what you want to call the text files to be saved
        :save_results: Boolean controls if things are saved to txt files.
            Currently saves scaled J and T
        :param index: the sideband index to which we want to normalize.
        :param saveFileName: Str of what you want to call the text files to be
            saved.
        :param scale_to_i: Boolean that controls to normalize to the ith
            sideband
            True -> Scale to ith | False -> scale to laser line

        returns: scaledJ, scaledT matrices scaled by Jxx strengths
        '''
        # Initialize the array for scaling
        Jxx_scales = np.array([])
        self.n_ref = index

        if scale_to_i:
            for idx in np.arange(len(Jxx[:, 0])):
                if Jxx[idx, 0] == index:
                    scale_to = Jxx[idx, 3]
                    print('scale to:', scale_to)
                    # sets the scale_to to be Jxx for the ith sideband
        else:
            scale_to = 1  # just makes this 1 if you don't want to scale to i

        scaledJ = Jraw  # initialize the scaled J matrix

        for idx in np.arange(len(Jxx[:, 0])):
            if Jxx[idx, 0] in observedSidebands:
                Jxx_scales = np.append(Jxx_scales, Jxx[idx, 3]/scale_to)
                print('Scaling sb order', Jxx[idx, 0])
                # Creates scaling factor

        for idx in np.arange(len(Jxx_scales)):
            scaledJ[:, :, idx] = Jraw[:, :, idx]*Jxx_scales[idx]
            # For each sideband scales Jraw by Jxx_scales

        scaledT = makeT(scaledJ, crystalAngle)
        # Makes scaledT from our new scaledJ

        if save_results:
            saveT(
                scaledJ, observedSidebands,
                "{}_scaledJMatrix.txt".format(saveFileName))
            saveT(
                scaledT, observedSidebands,
                "{}_scaledTMatrix.txt".format(saveFileName))
            # Saves the matrices

        return scaledJ, scaledT

    def Q_normalized_integrals(self, gamma1, gamma2, n, phi, beta):
        '''
        Returns Q_n^{HH}/Q_n^{LH} == Integrand_n^{HH}/Integrand_n^{LH}
        Unlike the normallized integrals used in early 2020 analysis, these
        integrals are of a given Fourier component's intensity from either the
        HH or LH band, and thus there is no prefactor related to the energy of
        the given sideband photon

        Parameters:
        :dephase: dephasing rate passed to intiallized TMAtrix object
        :w: the frequency of the THz field, in GHz
        :F: THz field strength in V/m
        :gamma1: Gamma1 parameter from Luttinger Hamiltonian
        :gamma2: Gamma2 parameter from Luttinger Hamiltonian
        :n: Order of the sideband for this integral
        :phi: [100] to THz orientation, passed from cost function function
            (in radians)
        :beta: experimentally measured g3/g2 ratio

        Returns: QRatio, the ratio of Q_n^{HH}/Q_n^{LH}
        '''

        mu_p, mu_m = self.mu_generator(gamma1, gamma2, phi, beta)
        w = self.Thz_w
        # Field = self.F
        hbar = self.hbar
        dephase = self.dephase
        int_cutoff = hbar*w/dephase*10

        # Because the integral is complex, the real and imaginary parts have to
        # be counted seperatly.

        re_Q_HH = intgt.quad(lambda x: np.real(self.Qintegrand(x, mu_p, n)),
                             0, int_cutoff, limit=1000000)[0]
        re_Q_LH = intgt.quad(lambda x: np.real(self.Qintegrand(x, mu_m, n)),
                             0, int_cutoff, limit=1000000)[0]
        im_Q_HH = intgt.quad(lambda x: np.imag(self.Qintegrand(x, mu_p, n)),
                             0, int_cutoff, limit=1000000)[0]
        im_Q_LH = intgt.quad(lambda x: np.imag(self.Qintegrand(x, mu_m, n)),
                             0, int_cutoff, limit=1000000)[0]

        # Combine the real and imaginary to have the full integral

        int_HH = re_Q_HH + 1j*im_Q_HH
        int_LH = re_Q_LH + 1j*im_Q_LH

        QRatio = int_HH/int_LH

        return QRatio

    def normalized_integrals(self, gamma1, gamma2, n, n_ref, phi, beta):
        '''
        Returns the plus and minus eta for a given sideband order, normalized
        to order n_ref (should probably be 10?). This whole calculation relies
        on calculating the ratio of these quantities to get rid of some
        troubling constants. So you need a reference integral.

        eta(n)+- =
        (w_nir + 2*n*w_thz)^2/(w_nir + 2*n_ref*w_thz)^2 *
        (mu_+-/mu_ref)^2 * (int(n)+-)^2/(int(n_ref)+)^2

        This takes gamma1 and gamma2 and gives the effective mass via
        mu_generator. It then calculates the normalized integrals for both mu's
        and gives eta, which is the integrals squared with some prefactors.
        Then you feed this into a cost function that varies gamma1 and gamma2.

        Parameters:
        :dephase: dephasing rate. Should be a few meV, ~the width of the
            exciton absorption peak (according to Qile). Should be float
        :lambda_nir: wavelength of NIR in nm
        :w_thz: frequency in GHz of fel. DO NOT give in angular form, the code
            does that for you.
        :F: THz field strength
        :gamma1: Gamma1 parameter in the luttinger hamiltonian.
            Textbook value of 6.85
        :gamma2: Gamma2 parameter in the luttinger hamiltonian.
            Textbook value of 2.1
        :n: Order of sideband for this integral
        :n_ref: Order of the reference integral which everything will
            be divided by
        :phi: [100] to THz orientation, passed from the data array
        :beta: experimentally measured g3/g2 ratio

        Returns: eta_p, eta_m the values of the eta parameter normalized to the
            appropriate sideband order for plus and minus values of mu.
        '''

        mu_p, mu_m = self.mu_generator(gamma1, gamma2, phi, beta)
        # gets the plus/minus effective mass
        omega_thz = self.Thz_w  # FEL frequency

        omega_nir = 2.998*10**8/(self.nir_wl) * 2 * np.pi
        # NIR frequency, takes nm (wavelength) and gives angular Hz

        # Field = self.F  # THz field

        hbar = self.hbar
        dephase = self.dephase
        int_cutoff = hbar*omega_thz/dephase*10
        # This cuts off the integral when x* dephase/hbaromega = 10
        # Therefore the values of the integrand will be reduced by a value
        # of e^(-10) which is about 4.5*10^(-5)

        re_int_ref = intgt.quad(lambda x: np.real(self.integrand(
            x, mu_p, n_ref)), 0, int_cutoff, limit=1000000)[0]
        re_int_p = intgt.quad(lambda x: np.real(self.integrand(
            x, mu_p, n)), 0, int_cutoff, limit=1000000)[0]
        re_int_m = intgt.quad(lambda x: np.real(self.integrand(
            x, mu_m, n)), 0, int_cutoff, limit=1000000)[0]
        '''
        Ok so these integrands are complex valued, but the intgt.quad
          integration does not work with that. So we split the integral up into
          two parts, real and imaginary parts. These lines calculate the real
          part for the reference, plus, and minus integrals.
        The integrals currently are limited to 10,000 iterations. No clue if
          that's a good amount or what. We could potentially make this simpler
          by doing a trapezoidal rule.
        We define the lambda function here to set all the values of the
          integrand function we want except for the variable of integration x
        '''

        im_int_ref = intgt.quad(lambda x: np.imag(self.integrand(
            x, mu_p, n_ref)), 0, int_cutoff, limit=1000000)[0]
        im_int_p = intgt.quad(lambda x: np.imag(self.integrand(
            x, mu_p, n)), 0, int_cutoff, limit=1000000)[0]
        im_int_m = intgt.quad(lambda x: np.imag(self.integrand(
            x, mu_m, n)), 0, int_cutoff, limit=1000000)[0]
        # Same as above but these are the imaginary parts of the integrals.

        int_ref = re_int_ref + 1j*im_int_ref
        int_p = re_int_p + 1j*im_int_p
        int_m = re_int_m + 1j*im_int_m
        # All the king's horses and all the king's men putting together our
        # integrals again. :)

        prefactor = (
            (omega_nir+2*n*omega_thz)**2)/((omega_nir+2*n_ref*omega_thz)**2)
        # This prefactor is the ratio of energy of the nth sideband to the
        # reference

        m_pre = (mu_m/mu_p)**2
        # There is a term of mu/mu_ref in the eta expression. For the

        eta_p = prefactor*(np.abs(int_p)**2)/(np.abs(int_ref)**2)
        eta_m = prefactor*m_pre*(np.abs(int_m)**2)/(np.abs(int_ref)**2)
        # Putting everthing together in one tasty little result

        return eta_p, eta_m

    def cost_func(
        self, gamma1, gamma2, observedSidebands, n_ref, Jexp, phi, beta,
        gc_fname, eta_folder
            ):
        '''
        This will sum up a cost function that takes the difference between
        the theory generated eta's and experimental scaled matrices

        eta+/eta+_ref = |Jxx|^2
        eta-/eta+_ref = |Jyy-Jxx/4|^2/|3/4|^2

        The cost function is given as

        Sqrt(|eta+(theory)-eta+(experiment)|^2+
            |eta-(theory)-eta-(experiment)|^2)

        Where the J elements have been scaled to the n_ref sideband (Jxx_nref)
        This is designed to run over and over again as you try different
        gamma values. On my (Joe) lab computer a single run takes ~300-400 sec.

        The function keeps track of values by writing a file with iteration,
        gamma1, gamma2, and cost for each run. This lets you keep track of the
        results as you run.

        Parameters:
        :dephase: dephasing rate. Should be a few meV, ~the width of the
            exciton absorption peak (according to Qile). Should be float
        :lambda_nir: wavelength of NIR in nm
        :w_thz: frequency of fel
        :F: THz field strength in kV/cm
        :gamma1: Gamma1 parameter in the luttinger hamiltonian.
            Textbook value of 6.85
        :gamma2: Gamma2 parameter in the luttinger hamiltonian.
            Textbook value of 2.1
        :n_ref: Order of the reference integral which everything will be
            divided by
        :Jexp: Scaled experimental Jones matrices in xy basis that will be
            compared to the theoretical values. Pass in the not flattened way.
        :phi: [100] to THz orientation, passed from the data array
        :beta: experimentally measured g3/g2 ratio
        :gc_fname: File name for the gammas and cost results
        :eta_folder: Folder name for the eta lists to go in
        :i: itteration, for parallel processing output purposes

        Returns:
        :costs: Cumulative cost function for that run
        :i: itteration, for parallel processing output purposes
        :eta_list: list of eta for's for each sideband order of the form

        sb order | eta_plus theory | eta_plus experiment | eta_minus thoery
            | eta_minus experiment
        .
        .
        .

        '''

        costs = 0  # initialize the costs for this run
        t_start = time.time()  # keeps track of the time the run started.
        eta_list = np.array([0, 0, 0, 0, 0])

        # dephase = self.dephase
        # lambda_nir = self.nir_wl
        omega_nir = 2.998*10**8/(self.nir_wl)*2*np.pi
        w_thz = self.Thz_w
        # F = self.F

        for idx in np.arrange(len(observedSidebands)):
            n = observedSidebands[idx]
            eta_p, eta_m = self.normalized_integrals(
                gamma1, gamma2, n, n_ref, phi, beta)
            # calculates eta from the normalized_integrals function
            prefactor = (
                (omega_nir+2*n*w_thz)**2)/((omega_nir+2*n_ref*w_thz)**2)
            # Have to hard code the index of the 16th order sideband
            #  (8,10,12,14,16)
            exp_p = prefactor*np.abs(Jexp[0, 0, idx])**2
            exp_m = prefactor*np.abs(
                Jexp[1, 1, idx]-(1/4)*Jexp[0, 0, idx])**2*(9/16)
            # calculates the experimental plus and minus values
            # 1/9/20 added prefactor to these bad boys

            costs += np.sqrt(np.abs((exp_p-eta_p)/(exp_p))**2 + np.abs(
                (exp_m-eta_m)/(exp_m))**2)
            # Adds the cost function for this sideband to the overall
            #   cost function
            # 01/08/20 Changed cost function to be the diference of the ratio
            #   of the two etas
            # 01/30/20 Changed cost function to be relative difference of
            #   eta_pm

            this_etas = np.array([n, eta_p, exp_p, eta_m, exp_m])
            eta_list = np.vstack((eta_list, this_etas))

        self.iterations += 1
        # Ups the iterations counter

        g1rnd = round(gamma1, 3)
        g2rnd = round(gamma2, 3)
        costs_rnd = round(costs, 5)
        # Round gamma1,gamma2,costs to remove float rounding bullshit

        g_n_c = str(self.iterations)+','+str(g1rnd)+','+str(g2rnd)+','+str(
            costs)+'\n'
        # String version of iteration, gamma1, gamma2, cost with a new line
        gc_file = open(gc_fname, 'a')
        # opens the gamma/cost file in append mode
        gc_file.write(g_n_c)  # writes the new line to the file
        gc_file.close()  # closes the file

        etas_header = "#\n"*95
        etas_header += f'# Dephasing: {self.dephase/(1.602*10**(-22))} eV \n'
        etas_header += f'# Detuning: {self.detune/(1.602*10**(-22))} eV \n'
        etas_header += f'# Field Strength: {self.F/(10**5)} kV/cm \n'
        etas_header += (f'# THz Frequency: '
                         f'{self.Thz_w/(10**9 * 2*np.pi)} GHz \n')
        etas_header += f'# NIR Wavelength: {self.nir_wl/(10**(-9))} nm \n'
        etas_header += ('sb order, eta_plus theory, eta_plus experiment,'
                        'eta_minus thoery, eta_minus experiment \n')
        etas_header += 'unitless, unitless, unitless, unitless, unitless \n'
        # Creates origin frienldy header for the eta's

        # eta_fname = 'eta_g1_' + str(g1rnd) + '_g2_' + str(g2rnd) + r'.txt'
        eta_fname = f'eta_g1_{g1rnd}_g2_{g2rnd}.txt'
        eta_path = os.path.join(eta_folder, eta_fname)
        # creates the file for this run of etas

        eta_list = eta_list[1:,:]
        np.savetxt(eta_path, eta_list, delimiter=',',
                   header=etas_header, comments='')
        # save the etas for these gammas

        t_taken = round(time.time()-t_start, 5)
        # calcuates time taken for this run

        print("  ")
        print("--------------------------------------------------------------")
        print("  ")
        print(f'Iteration number {self.iterations} / {self.max_iter} done')
        print('for gamma1, gamma2 = ', g1rnd, g2rnd)
        print('Cost function is = ', costs_rnd)
        print('This calculation took ', t_taken, ' seconds')
        print("  ")
        print("--------------------------------------------------------------")
        print("  ")
        # These print statements help you keep track of what's going on as this
        #   goes on and on and on.


        return costs

    def Q_cost_func(
                    self, gamma1, gamma2, n, Texp, crystalAngles, beta,
                    gc_fname, Q_folder):
        '''
        This compairs the T Matrix components measured by experiment to the

        '''
        costs = 0  # Initialize the costs
        t_start = time.time()
        Q_list = np.array([0, 0, 0, 0, 0])

        # dephase = self.dephase
        # w = self.Thz_w
        # F = self.F

        for idx in np.arange(len(crystalAngles)):
            phi = float(crystalAngles[idx])
            phi_rad = phi*np.pi/180
            theta = phi_rad + np.pi/4
            # Calculate the Theoretical Q Ratio
            QRatio = self.Q_normalized_integrals(
                                                 gamma1, gamma2, n, phi_rad,
                                                 beta)
            # Prefactor for experimental T Matirx algebra
            PHI = 5/(3*(np.sin(2*theta) - 1j*beta*np.cos(2*theta)))
            THETA = 1/(np.sin(2*theta)-1j*beta*np.cos(2*theta))
            ExpQ = (Texp[idx, 0, 0]+PHI*Texp[idx, 0, 1])/(
                    Texp[idx, 0, 0]-THETA*Texp[idx, 0, 1])

            costs += np.abs((ExpQ - QRatio)/QRatio)

            this_Qs = np.array([n, np.real(ExpQ), np.imag(ExpQ),
                                np.real(QRatio), np.imag(QRatio)])
            Q_list = np.vstack((Q_list, this_Qs))

        self.iterations += 1

        g1rnd = round(gamma1, 3)
        g2rnd = round(gamma2, 3)
        costs_rnd = round(costs, 5)

        g_n_c = (str(self.iterations)+','+str(g1rnd)+','+str(g2rnd)+','+
                 str(costs)+'\n')
        gc_file = open(gc_fname, 'a')
        gc_file.write(g_n_c)
        gc_file.close()

        # Origin Header
        Q_header = "#\n"*95
        Q_header += f'# Dephasing: {self.dephase/(1.602*10**(-22))} eV \n'
        Q_header += f'# Detuning: {self.detune/(1.602*10**(-22))} eV \n'
        Q_header += f'# Field Strength: {self.F/(10**5)} kV/cm \n'
        Q_header += f'# THz Frequncy {self.Thz_w/(10**9 *2*np.pi)} GHz \n'
        Q_header += f'# NIR Wavelength {self.nir_wl/(10**(-9))} nm \n'
        Q_header += ('sb order, QRatio Experiment Real, Imaginary,'
                     'QRatio Theory Real, Imaginary \n')
        Q_header += 'unitless, unitless, unitless, unitless, unitless \n'

        # Eta File Name
        Q_fname = f'Q_g1_{g1rnd}_g2_{g2rnd}.txt'
        Q_path = os.path.join(Q_folder, Q_fname)

        Q_list = Q_list[1:, :]
        np.savetxt(Q_path, Q_list, delimiter=',',
                   header=Q_header, comments='')

        t_taken = round(time.time() - t_start,5)

        print("  ")
        print("--------------------------------------------------------------")
        print("  ")
        print(f'Iteration number {self.iterations} / {self.max_iter} done')
        print('for gamma1, gamma2 = ', g1rnd,g2rnd)
        print('Cost function is = ', costs_rnd)
        print('This calculation took ', t_taken, ' seconds')
        print("  ")
        print("--------------------------------------------------------------")
        print("  ")

        return costs

    def gamma_sweep(self, gamma1_array, gamma2_array, observedSidebands, n_ref,
                    Jexp, gc_fname, eta_folder, save_results=True):
        '''
        This function calculates the integrals and cost function for an array
        of gamma1 and gamma2. You can pass any array of gamma1 and gamma2
        values and this will return the costs for all those values. Let's you
        avoid the weirdness of fitting algorithims.

        Parameters:
        :dephase: dephasing rate. Should be a few meV, ~the width of the
            exciton absorption peak (according to Qile). Should be float
        :lambda_nir: wavelength of NIR in nm
        :w_thz: frequency of fel
        :F: THz field strength
        :gamma1: Gamma1 parameter in the luttinger hamiltonian.
            Textbook value of 6.85
        :gamma2: Gamma2 parameter in the luttinger hamiltonian.
            Textbook value of 2.1
        :n: Order of sideband for this integral
        :n_ref: Order of the reference integral which everything will be
            divided by
        :observedSidebands: List or array of observed sidebands. The code will
            loop over sidebands in this array.
        :Jexp: Scaled experimental Jones matrices in xy basis that will be
            compared to the theoretical values. Pass in the not flattened way.
        :gc_fname: File name for the gammas and cost functions, include .txt
        :eta_folder: Folder name for the eta lists to go in

        Returns: gamma_cost_array of form
        gamma1 | gamma2 | cost |
        .           .       .
        .           .       .
        .           .       .

        This is just running cost_func over and over again essentially.
        '''

        # dephase = self.dephase
        # lambda_nir = self.nir_wl
        # w_thz = self.Thz_w
        # F = self.F

        self.max_iter = len(gamma1_array)*len(gamma2_array)

        gamma_cost_array = np.array([0, 0, 0])
        # Initialize the gamma cost array

        gammas_costs = np.array([])
        # This is just for initializing the gamma costs file

        gammacosts_header = "#\n"*95
        gammacosts_header += (f'# Dephasing:'
                              f'{self.dephase/(1.602*10**(-22))} eV \n')
        gammacosts_header += (f'# Detuning: '
                              f'{self.detune/(1.602*10**(-22))} eV \n')
        gammacosts_header += f'# Field Strength: {self.F/(10**5)} kV/cm \n'
        gammacosts_header += (f'# THz Frequency: '
                              f'{self.Thz_w/(10**9 * 2*np.pi)} GHz \n')
        gammacosts_header += (f'# NIR Wavelength: '
                              f'{self.nir_wl/(10**(-9))} nm \n')
        gammacosts_header += 'Iteration, Gamma1, Gamma2, Cost Function \n'
        gammacosts_header += 'unitless, unitless, unitless, unitless \n'
        # Creates origin frienldy header for gamma costs

        np.savetxt(gc_fname, gammas_costs, delimiter=',',
                   header=gammacosts_header, comments='')
        # create the gamma cost file

        # data = [gamma1_array, gamma2_array]

        for gamma1 in gamma1_array:
            for gamma2 in gamma2_array:
                cost = self.cost_func(gamma1, gamma2, observedSidebands,
                                      n_ref, Jexp, gc_fname, eta_folder)
                this_costngamma = np.array([gamma1, gamma2, cost])
                gamma_cost_array = np.vstack((
                                            gamma_cost_array, this_costngamma))
                # calculates the cost for each gamma1/2 and adds the gamma1,
                #   gamma2, and cost to the overall array.

        # gamma_cost_array = gamma_cost_final[1:,:]

        # if save_results:
        #     sweepcosts_header = "#\n"*100
        #     sweepcosts_header += 'Gamma1, Gamma2, Cost Function \n'
        #     sweepcosts_header += 'unitless, unitless, unitless \n'
        #
        #     sweep_name = 'sweep_costs_' + gc_fname
        #     np.savetxt(sweep_name,gamma_cost_array,delimiter = ',',
        #         header = sweepcosts_header, comments = '')
        # Ok so right now I think I am going to get rid of saving this file
        #   since it has the same information as the file that is saved in
        #   cost_func but that file is updated every interation where this
        #   one only works at the end. So if the program gets interrupted
        #   the other one will still give you some information.

        return gamma_cost_array

    def gamma_th_sweep(self, gamma1_array, gamma2_array, n, crystalAngles,
                       Texp, gc_fname, Q_folder, save_results=True):
        '''
        This function calculates the integrals and cost function for an array
        of gamma1 and gamma2. You can pass any array of gamma1 and gamma2
        values and this will return the costs for all those values. Let's you
        avoid the weirdness of fitting algorithims.

        Parameters:
        :dephase: dephasing rate. Should be a few meV, ~the width of the
            exciton absorption peak (according to Qile). Should be float
        :lambda_nir: wavelength of NIR in nm
        :w_thz: frequency of fel
        :F: THz field strength
        :gamma1: Gamma1 parameter in the luttinger hamiltonian.
            Textbook value of 6.85
        :gamma2: Gamma2 parameter in the luttinger hamiltonian.
            Textbook value of 2.1
        :n: Order of sideband for this integral
        :crystalAngles: List or array of crystal Angles. The code will
            loop over sidebands in this array.
        :TExp: Scaled experimental Jones matrices in xy basis that will be
            compared
            to the theoretical values. Pass in the not flattened way.
        :gc_fname: File name for the gammas and cost functions, include .txt
        :Q_folder: Folder name for the eta lists to go in

        Returns: gamma_cost_array of form
        gamma1 | gamma2 | cost |
        .           .       .
        .           .       .
        .           .       .

        This is just running cost_func over and over again essentially.
        '''
        # Hard Coding the experimental g3/g2 factor
        beta = 1.42
        # dephase = self.dephase
        # lambda_nir = self.nir_wl
        # w_thz = self.Thz_w
        # F = self.F

        self.iterations = 0
        self.max_iter = len(gamma1_array)*len(gamma2_array)

        gamma_cost_array = np.array([0, 0, 0])
        # Initialize the gamma cost array

        gammas_costs = np.array([])
        # This is just for initializing the gamma costs file

        gammacosts_header = "#\n"*95
        gammacosts_header += (f'# Dephasing:'
                              f'{self.dephase/(1.602*10**(-22))} eV \n')
        gammacosts_header += (f'# Detuning: '
                              f'{self.detune/(1.602*10**(-22))} eV \n')
        gammacosts_header += f'# Field Strength: {self.F/(10**5)} kV/cm \n'
        gammacosts_header += (f'# THz Frequency: '
                              f'{self.Thz_w/(10**9 * 2*np.pi)} GHz \n')
        gammacosts_header += (f'# NIR Wavelength: '
                              f'{self.nir_wl/(10**(-9))} nm \n')
        gammacosts_header += 'Iteration, Gamma1, Gamma2, Cost Function \n'
        gammacosts_header += 'unitless, unitless, unitless, unitless \n'
        # Creates origin frienldy header for gamma costs

        np.savetxt(gc_fname, gammas_costs, delimiter=',',
                   header=gammacosts_header, comments='')
        # create the gamma cost file

        # data = [gamma1_array, gamma2_array]

        for gamma1 in gamma1_array:
            for gamma2 in gamma2_array:
                cost = self.Q_cost_func(gamma1, gamma2, n,
                                        Texp, crystalAngles, beta,
                                        gc_fname, Q_folder)
                this_costngamma = np.array([gamma1, gamma2, cost])
                gamma_cost_array = np.vstack((
                                            gamma_cost_array, this_costngamma))
                # calculates the cost for each gamma1/2 and adds the gamma1,
                #   gamma2, and cost to the overall array.

        # gamma_cost_array = gamma_cost_final[1:,:]

        # if save_results:
        #     sweepcosts_header = "#\n"*100
        #     sweepcosts_header += 'Gamma1, Gamma2, Cost Function \n'
        #     sweepcosts_header += 'unitless, unitless, unitless \n'
        #
        #     sweep_name = 'sweep_costs_' + gc_fname
        #     np.savetxt(sweep_name,gamma_cost_array,delimiter = ',',
        #         header = sweepcosts_header, comments = '')
        # Ok so right now I think I am going to get rid of saving this file
        #   since it has the same information as the file that is saved in
        #   cost_func but that file is updated every interation where this
        #   one only works at the end. So if the program gets interrupted
        #   the other one will still give you some information.

        return gamma_cost_array
