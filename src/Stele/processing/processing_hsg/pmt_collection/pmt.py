import json


class PMT(object):
    def __init__(self, file_name):
        """
        Initializes a SPEX spectrum.  It'll open a file, and bring in the
        details of a sideband spectrum into the object.  There isn't currently
        any reason to use inheritance here, but it could be extended later to
        include PLE or something of the sort.

        attributes:
            self.parameters - dictionary of important experimental parameters
                              this will not necessarily be the same for each
                              file in the object
            self.fname - the current file path

        :param file_name: The name of the PMT file
        :type file_name: str
        :return: None
        """
        # print "This started"
        self.fname = file_name
        # self.files_included = [file_name]
        with open(file_name, 'r') as f:
            param_str = ''
            # Needed to move past the first line, which is the sideband order.
            # Not generally useful
            line = f.readline()
            line = f.readline()
            while line[0] == '#':
                param_str += line[1:]
                line = f.readline()

            self.parameters = json.loads(param_str)
