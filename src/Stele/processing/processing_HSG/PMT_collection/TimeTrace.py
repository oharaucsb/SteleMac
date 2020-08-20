from .PMT_collection.pmt import PMT
from .PMT_collection.HighSidebandPMT import HighSidebandPMT


class TimeTrace(PMT):
    """
    This class will be able to handle time traces output by the PMT softare.
    """
    def __init__(self, file_path):
        super(HighSidebandPMT, self).__init__(file_path)
