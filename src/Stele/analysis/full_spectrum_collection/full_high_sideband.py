import os
import errno
import json
import numpy as np
from .full_spectrum_collection.full_spectrum import FullSpectrum
from .full_spectrum_collection import helper_functions as helpers

np.set_printoptions(linewidth=500)


# One of the main results is the HighSidebandCCD.sb_results array. These are
# the various mappings between index and real value I deally, this code should
# be converted to pandas to avoid this issue,
# but that's outside the scope of current work.
# [sb number, Freq (eV), Freq error (eV), Gauss area (arb.), Area error,
#    Gauss linewidth (eV), Linewidth error (eV)]
# [    0    ,      1   ,        2,      ,        3         ,
#         4    ,         5           ,        6            ]
class sbarr(object):
    SBNUM = 0
    CENFREQ = 1
    CENFREQERR = 2
    AREA = 3
    AREAERR = 4
    WIDTH = 5
    WIDTHERR = 6


class FullHighSideband(FullSpectrum):
    """
    I'm imagining this class is created with a base CCD file, then gobbles up
    other spectra that belong with it, then grabs the PMT object to normalize
    everything, assuming that PMT object exists.
    """

    def __init__(self, initial_CCD_piece):
        """
        Initialize a full HSG spectrum.  Starts with a single CCD image, then
        adds more on to itself using stitch_hsg_dicts.

        Creates:
        self.fname = file name of the initial_CCD_piece
        self.sb_results = The sideband details from the initializing data
        self.parameters = The parameter dictionary of the initializing data.
                    May not have all details of spectrum pieces added later.
        self.full_dict = a copy of the sb_results without the zeroth column,
                    which is SB order

        :param initial_CCD_piece: The starting part of the spectrum, often the
            lowest orders seen by CCD
        :type initial_CCD_piece: HighSidebandCCD
        :return: None
        """
        self.fname = initial_CCD_piece.fname
        try:
            self.sb_results = initial_CCD_piece.sb_results
        except AttributeError:
            print(initial_CCD_piece.full_dict)
            raise
        self.parameters = initial_CCD_piece.parameters
        self.parameters['files_here'] = [
            initial_CCD_piece.fname.split('/')[-1]]
        self.full_dict = {}
        for sb in self.sb_results:
            self.full_dict[sb[0]] = np.asarray(sb[1:])

    @staticmethod
    def parse_sb_array(arr):
        """
        Check to make sure the first even order sideband in an array is not
        weaker than the second even order. If this happens, it's likely because
        the SB was in the short pass filter and isn't work counting.

        We cut it out to prevent it from itnerfering with calculating overlaps
        :param arr:
        :return:
        """
        arr = np.array(arr)

        # make sure they're both pos
        if (arr[0, sbarr.SBNUM] > 0 and arr[1, sbarr.SBNUM] > 0 and
            # and the fact the area is less
                arr[0, sbarr.AREA] < arr[1, sbarr.AREA]):
            # print "REMOVING FIRST SIDEBAND FROM FULLSIDEBAND"
            # print arr[0]
            # print arr[1]
            arr = arr[1:]

        full_dict = {}
        for sb in arr:
            full_dict[sb[0]] = np.asarray(sb[1:])
        return full_dict, arr

    def add_CCD(self, ccd_object, verbose=False, force_calc=None, **kwargs):
        """
        This method will be called by the stitch_hsg_results function to add
            another CCD image to the spectrum.

        :param ccd_object: The CCD object that will be stiched into the current
            FullHighSideband object
        :type ccd_object: HighSidebandCCD
        :return: None
        """
        if self.parameters["gain"] == ccd_object.parameters["gain"]:
            calc = False
        else:
            calc = True
        if force_calc is not None:
            calc = force_calc
        # cascading it through, starting to think
        if "need_ratio" in kwargs:
            # everything should be in a kwarg
            calc = kwargs.pop("need_ratio")
        try:
            # self.full_dict = stitch_hsg_dicts(
            #    self.full_dict, ccd_object.full_dict, need_ratio=calc,
            #    verbose=verbose)
            self.full_dict = helpers.stitch_hsg_dicts(
                self, ccd_object, need_ratio=calc, verbose=verbose, **kwargs)
            self.parameters['files_here'].append(
                ccd_object.fname.split('/')[-1])
            # update sb_results, too
            sb_results = [[k]+list(v) for k, v in list(self.full_dict.items())]
            sb_results = np.array(sb_results)
            self.sb_results = sb_results[sb_results[:, 0].argsort()]
        except AttributeError:
            print('Error, not enough sidebands to fit here! {}, {}, {}, {}'.format(
                    self.parameters["series"],
                    self.parameters["spec_step"],
                    ccd_object.parameters["series"],
                    ccd_object.parameters["spec_step"]))

    def add_PMT(self, pmt_object, verbose=False):
        """
        This method will be called by the stitch_hsg_results function to add
        the PMT data to the spectrum.
        """
        # print "I'm adding PMT once"
    # self.full_dict = stitch_hsg_dicts(pmt_object.full_dict, self.full_dict,
    # need_ratio=True, verbose=False)
        self.full_dict = helpers.stitch_hsg_dicts(
            pmt_object, self, need_ratio=True, verbose=verbose)
        # if verbose:
        #     self.full_dict, ratio = self.full_dict
        # print "I'm done adding PMT data"
        self.parameters['files_here'].append(
            pmt_object.parameters['files included'])
        self.make_results_array()
        # if verbose:
        #     return ratio

    def make_results_array(self):
        """
        The idea behind this method is to create the sb_results array from the
        finished full_dict dictionary.
        """
        self.sb_results = None
        # print "I'm making the results array:", sorted(self.full_dict.keys())
        for sb in sorted(self.full_dict.keys()):
            # print "Going to add this", sb
            try:
                # too many parenthesis?
                self.sb_results = np.vstack(
                    (self.sb_results, np.hstack((sb, self.full_dict[sb]))))
            except ValueError:
                # print "It didn't exist yet!"
                self.sb_results = np.hstack((sb, self.full_dict[sb]))
                # print "and I made this array:", self.sb_results[:, 0]

    def save_processing(
     self, file_name, folder_str, marker='', index='', verbose=''):
        """
        This will save all of the self.proc_data and the results from the
        fitting of this individual file.

        Format:
        fit_fname = file_name + '_' + marker + '_' + str(index) + '_full.txt'

        Inputs:
        file_name = the beginning of the file name to be saved
        folder_str = the location of the folder where the file will be saved,
                     will create the folder, if necessary.
        marker = I...I don't know what this was originally for
        index = used to keep these files from overwriting themselves when in a
                list

        Outputs:
        Two files, one that is self.proc_data, the other is self.sb_results
        """
        try:
            os.mkdir(folder_str)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        temp = np.array(self.sb_results)

        ampli = np.array([temp[:, 3] / temp[:, 5]])  # I'm pretty sure this is
        # amplitude, not area
        temp[:, 5:7] = temp[:, 5:7] * 1000  # For meV linewidths
        if verbose:
            print("sb_results", self.sb_results.shape)
            print("ampli", ampli.shape)
        save_results = np.hstack((temp, ampli.T))

        # spectra_fname = file_name + '_' + marker + '_' + str(index) + '.txt'
        fit_fname = file_name + '_' + marker + '_' + str(index) + '_full.txt'
        # self.save_name = spectra_fname

        # self.parameters['addenda'] = self.addenda
        # self.parameters['subtrahenda'] = self.subtrahenda
        try:
            # PMT files add unnecessary number of lines, dump it into one line
            # by casting it to a string.
            reduced = self.parameters.copy()
            reduced["files_here"] = str(reduced["files_here"])
            parameter_str = json.dumps(
                reduced, sort_keys=True, indent=4, separators=(',', ': '))
        except Exception as e:
            print(e)
            print("Source: EMCCD_image.save_images\nJSON FAILED")
            print("Here is the dictionary that broke JSON:\n", self.parameters)
            return
        parameter_str = parameter_str.replace('\n', '\n#')

        # Make the number of lines constant so importing is easier
        num_lines = parameter_str.count('#')
        # for num in range(99 - num_lines): parameter_str += '\n#'
        parameter_str += '\n#' * (99 - num_lines)
        # origin_import_spec = '\nNIR frequency,Signal,Standard error' +
        #   '\neV,arb. u.,arb. u.'
        # spec_header = '#' + parameter_str + '\n#' +
        #    self.description[:-2] + origin_import_spec

        origin_import_fits = '\nSideband,Center energy,error,Sideband strength,error,Linewidth,error,Amplitude'+\
                             '\norder,eV,,arb. u.,,meV,,arb. u.\n' + ','.join([marker]*8)
        fits_header = '#' + parameter_str + origin_import_fits

        # np.savetxt(os.path.join(folder_str, spectra_fname), self.proc_data,
        #            delimiter=',', header=spec_header, comments='', fmt='%f')
        np.savetxt(os.path.join(folder_str, fit_fname), save_results,
                   delimiter=',', header=fits_header, comments='', fmt='%0.6e')

        if verbose:
            print("Save image.\nDirectory: {}".format(
                os.path.join(folder_str, fit_fname)))
