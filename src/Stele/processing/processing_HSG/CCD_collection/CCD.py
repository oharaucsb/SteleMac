import json
import numpy as np
import os

np.set_printoptions(linewidth=500)


class CCD(object):
    def __init__(self, fname, spectrometer_offset=None):
        """
        This will read the appropriate file and make a basic CCD object.
        Fancier things will be handled with the sub classes.

        Creates:
        self.parameters = Dictionary holding all of the information from the
                          data file, which comes from the JSON encoded header
                          in the data file.
        self.description = string that is the text box from data taking GUI
        self.raw_data = raw data output by measurement software, wavelength vs.
                        data, errors. There may be text for some of the entries
                        corresponding to text used for Origin imports, but they
                        should appear as np.nan
        self.ccd_data = semi-processed 1600 x 3 array of photon energy vs. data
                        with standard error of mean at that pixel calculated by
                        taking multiple images. Standard error is calculated
                        from the data collection software

        Most subclasses should make a self.proc_data, which will do whatever
        processing is required to the ccd_data, such as normalizing, taking
        ratios, etc.

        :param fname: file name where the data is saved
        :type fname: str
        :param spectrometer_offset: if the spectrometer won't go where it's
            told, use this to correct the wavelengths (nm)
        :type spectrometer_offset: float
        """

        self.fname = fname

        # Checking restrictions from Windows path length limits. Check if able
        # open the file:
        try:
            with open(fname) as f:
                pass
        except FileNotFoundError:
            # Couldn't find the file. Could be you passed the wrong one, but
            # I'm finding with a large number of subfolders for polarimetry
            # stuff, you end up exceeding Windows'  filelength limit. Haven't
            # tested on Mac or UNC moutned drives (e.g \\128.x.x.x\Sherwin\)
            fname = r"\\?\\" + os.path.abspath(fname)

        # Read in the JSON-formatted parameter string.
        # The lines are all prepended by '#' for easy numpy importing
        # so loop over all those lines
        with open(fname, 'r') as f:
            param_str = ''
            line = f.readline()
            while line[0] == '#':
                # changed 09/17/18
                # This line assumed there was a single '#'
                # param_str += line[1:]
                # while this one handles everal (because I found old files
                # which had '## <text>...'
                param_str += line.replace("#", "")
                line = f.readline()
            # Parse the JSON string
            try:
                self.parameters = json.loads(param_str)
            except json.JSONDecodeError:
                # error from _really_ old data where comments were dumped after
                # a single-line json dumps
                self.parameters = json.loads(param_str.splitlines()[0])

        # Spec[trometer] steps are set to define the same physical data, but
        # taken at different spectrometer center wavelengths. This value is
        # used later for stitching these scans together
        try:
            self.parameters["spec_step"] = int(self.parameters["spec_step"])
        except (ValueError, KeyError):
            # If there isn't a spe
            self.parameters["spec_step"] = 0

        # Slice through 3 to get rid of comments/origin info. Would likely be
        # better to check np.isnan() and slicing out those nans. I used flipup
        # so that the x-axis is an increasing function of frequency
        self.raw_data = np.flipud(np.genfromtxt(
            fname, comments='#', delimiter=',')[3:])

        # The camera chip is 1600 pixels wide. This line was redudent with the
        # [3:] slice above and served to make sure there weren't extra stray
        # bad lines hanging around.
        #
        # This should also be updated some day to compensate for any horizontal
        # bining on the chip, or masking out points that are bad (cosmic ray
        # making it through processing, room lights or monitor lines
        # interfering with signal)
        self.ccd_data = np.array(self.raw_data[:1600, :])

        # Check to see if the spectrometer offset is set. This isn't specified
        # during data collection. This is a value that can be appended
        # when processing if it's realized the data is offset.
        # This allows the offset to be specified and kept with the data file itself,
        # instead of trying to do it in individual processing scripts
        #
        # It's allowed as a kwarg parameter in this script for trying to determine
        # what the correct offset should be
        if spectrometer_offset is not None or "offset" in self.parameters:
            try:
                self.ccd_data[:, 0] += float(self.parameters["offset"])
            except:
                self.ccd_data[:, 0] += spectrometer_offset

        # Convert from nm to eV
        # self.ccd_data[:, 0] = 1239.84 / self.ccd_data[:, 0]
        self.ccd_data[:, 0] = photon_converter["nm"]["eV"](self.ccd_data[:, 0])
