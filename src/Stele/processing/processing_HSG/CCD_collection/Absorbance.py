import os
import errno
import json
import numpy as np

np.set_printoptions(linewidth=500)


class Absorbance(CCD):
    def __init__(self, fname):
        """
        There are several ways Absorbance data can be loaded
        You could try to load the abs data output from data collection directly,
        which has the wavelength, raw, blank and actual absorbance data itself.
        This is best way to do it.

        Alternatively, you could want to load the raw transmission/reference
        data, ignoring (or maybe not even having) the abs calculated
        from the data collection software. If you want to do it this way,
        you should pass fname as a list where the first element is the
        file name for the reference data, and the second is the absorbance data
        At first, it didn't really seem to make sense to let you pass just the
        raw reference or raw abs data,


        Creates:
        self.ref_data = np array of the reference,
            freq (eV) vs. reference (counts)
        self.raw_data = np.array of the raw absorption spectrum,
            freq (eV) vs. reference (counts)
        self.proc_data = np.array of the absorption spectrum
            freq (eV) vs. "absorbance" (dB)

        Note, the error bars for this data haven't been defined.

        :param fname: either an absorbance filename, or a length 2 list of filenames
        :type fname: str
        :return: None
        """
        if "abs_" in fname:
            super(Absorbance, self).__init__(fname)
            # Separate into the separate data sets
            #   The raw counts of the reference data
            self.ref_data = np.array(self.ccd_data[:, [0, 1]])
            #   Raw counts of the sample
            self.raw_data = np.array(self.ccd_data[:, [0, 2]])
            #   The calculated absorbance data (-10*log10(raw/ref))
            self.proc_data = np.array(self.ccd_data[:, [0, 3]]) # Already in dB's

        else:
            # Should be here if you pass the reference/trans filenames
            try:
                super(Absorbance, self).__init__(fname[0])
                self.ref_data = np.array(self.ccd_data)

                super(Absorbance, self).__init__(fname[1])
                self.raw_data = np.array(self.ccd_data)
            except ValueError:
                # ValueError gets thrown when importing older data
                # which had more headers than data columns. Enforce
                # only loading first two columns to avoid numpy trying
                # to parse all of the data

                # See CCD.__init__ for what's going on.

                self.ref_data = np.flipud(np.genfromtxt(fname[0], comments='#',
                                                        delimiter=',', usecols=(0, 1)))

                self.ref_data = np.array(self.ref_data[:1600, :])
                self.ref_data[:, 0] = 1239.84 / self.ref_data[:, 0]

                self.raw_data = np.flipud(np.genfromtxt(fname[1], comments='#',
                                                        delimiter=',', usecols=(0, 1)))

                self.raw_data = np.array(self.raw_data[:1600, :])
                self.raw_data[:, 0] = 1239.84 / self.raw_data[:, 0]
            except Exception as e:
                print("Exception opening absorbance data,", e)

            # Calculate the absorbance from the raw camera counts.
            self.proc_data = np.empty_like(self.ref_data)
            self.proc_data[:, 0] = self.ref_data[:, 0]
            self.proc_data[:, 1] = -10*np.log10(self.raw_data[:, 1] / self.ref_data[:,
                                                                     1])

    def abs_per_QW(self, qw_number):
        """

        :param qw_number: number of quantum wells in the sample.
        :type qw_number: int
        :return: None
        """
        """
        This method turns the absorption to the absorbance per quantum well.  Is
        that how this data should be reported?

        Also, I'm not sure if columns 1 and 2 are correct.
        """
        temp_abs = -np.log(self.proc_data[:, 1] / self.proc_data[:, 2]) / qw_number
        self.proc_data = np.hstack((self.proc_data, temp_abs))

    def fft_smooth(self, cutoff, inspectPlots=False):
        """
        This function removes the Fabry-Perot that affects the absorption data

        creates:
        self.clean = np.array of the Fourier-filtered absorption data, freq (eV) vs. absorbance (dB!)
        self.parameters['fourier cutoff'] = the low pass cutoff frequency, in eV**(-1)
        :param cutoff: Fourier frequency of the cut off for the low pass filter
        :type cutoff: int or float
        :param inspectPlots: Do you want to see the results?
        :type inspectPlots: bool
        :return: None
        """
        # self.fixed = -np.log10(abs(self.raw_data[:, 1]) / abs(self.ref_data[:, 1]))
        # self.fixed = np.nan_to_num(self.proc_data[:, 1])
        # self.fixed = np.column_stack((self.raw_data[:, 0], self.fixed))
        self.parameters['fourier cutoff'] = cutoff
        self.clean = low_pass_filter(self.proc_data[:, 0], self.proc_data[:, 1], cutoff, inspectPlots)

    def save_processing(self, file_name, folder_str, marker='', index=''):
        """
        This bad boy saves the absorption spectrum that has been manipulated.

        Saves 100 lines of comments.

        :param file_name: The base name of the file to be saved
        :type file_name: str
        :param folder_str: The name of the folder where the file will be saved
        :type folder_str: str
        :param marker: A further label that might be the series tag or something
        :type marker: str
        :param index: If multiple files are being saved with the same name, include an integer to append to the end of the file
        :type index: int
        :return: None
        """
        try:
            os.mkdir(folder_str)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        spectra_fname = file_name + '_' + marker + '_' + str(index) + '.txt'
        self.save_name = spectra_fname

        try:
            parameter_str = json.dumps(self.parameters, sort_keys=True, indent=4, separators=(',', ': '))
        except:
            print("Source: EMCCD_image.save_images\nJSON FAILED")
            print("Here is the dictionary that broke JSON:\n", self.parameters)
            return
        parameter_str = parameter_str.replace('\n', '\n#')

        num_lines = parameter_str.count('#')  # Make the number of lines constant so importing into Origin is easier
        # for num in range(99 - num_lines): parameter_str += '\n#'
        parameter_str += '\n#' * (99 - num_lines)

        origin_import_spec = '\nNIR frequency,Signal,Standard error\neV,arb. u.,arb. u.'
        spec_header = '#' + parameter_str + origin_import_spec
        # spec_header = '#' + parameter_str + '\n#' + self.description[:-2] + origin_import_spec

        np.savetxt(os.path.join(folder_str, spectra_fname), self.proc_data, delimiter=',',
                   header=spec_header, comments='', fmt='%0.6e')
        spectra_fname = 'clean ' + spectra_fname
        np.savetxt(os.path.join(folder_str, spectra_fname), self.clean, delimiter=',',
                   header=spec_header, comments='', fmt='%0.6e')
        print("Save image.\nDirectory: {}".format(os.path.join(folder_str, spectra_fname)))

# class LaserLineCCD(HighSidebandCCD):
#     """
#     Class for use when doing alinging/testing by sending the laser
#     directly into the CCD. Modifies how "sidebands" and guess and fit,
#     simply looking at the max signal.
#     """
#     def guess_sidebands(self, cutoff=8, verbose=False, plot=False):
#         pass
