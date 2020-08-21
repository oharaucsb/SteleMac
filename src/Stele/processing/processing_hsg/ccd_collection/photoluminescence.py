import numpy as np
from .CCD_collection import CCD


class Photoluminescence(CCD.CCD):
    def __init__(self, fname):
        """
        This object handles PL-type data. The only distinction from the parent
        class is that the CCD data gets normalized to the exposure time to make
        different exposures directly comparable.

        creates:
        self.proc_data = self.ccd_data divided by the exposure time
                         units: PL counts / second
        :param fname: name of the file
        :type fname: str

        """
        super(Photoluminescence, self).__init__(fname)

        # Create a copy of the array , and then normalize the signal and the
        #  errors by the exposure time
        self.proc_data = np.array(self.ccd_data)
        self.proc_data[:, 1] = (self.proc_data[:, 1]
                                / self.parameters['exposure'])
        self.proc_data[:, 2] = (self.proc_data[:, 2]
                                / self.parameters['exposure'])
