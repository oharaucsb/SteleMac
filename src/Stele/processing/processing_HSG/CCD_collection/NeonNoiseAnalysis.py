import os
import errno
import numpy as np

np.set_printoptions(linewidth=500)


class NeonNoiseAnalysis(CCD):
    """
    This class is used to make handling neon calibration lines easier.  It's not great.
    """
    def __init__(self, fname, spectrometer_offset=None):
        # print 'opening', fname
        super(NeonNoiseAnalysis, self).__init__(fname, spectrometer_offset=spectrometer_offset)

        self.addenda = self.parameters['addenda']
        self.subtrahenda = self.parameters['subtrahenda']

        self.noise_and_signal()
        self.process_stuff()

    def noise_and_signal(self):
        """
        This bad boy calculates the standard deviation of the space between the
        neon lines.

        The noise regions are, in nm:
        high: 784-792
        low1: 795-806
        low2: 815-823
        low3: 831-834

        the peaks are located at, in nm:
        #1, weak: 793.6
        #2, medium: 794.3
        #3, medium: 808.2
        #4, weak: 825.9
        #5, strong: 830.0
        """
        print('\n\n')

        self.ccd_data = np.flipud(self.ccd_data)
        # self.high_noise_region = np.array(self.ccd_data[30:230, :])
        self.high_noise_region = np.array(self.ccd_data[80:180, :])  # for dark current measurements
        self.low_noise_region1 = np.array(self.ccd_data[380:700, :])
        self.low_noise_region2 = np.array(self.ccd_data[950:1200, :])
        self.low_noise_region3 = np.array(self.ccd_data[1446:1546, :])

        # self.high_noise = np.std(self.high_noise_region[:, 1])
        self.high_noise_std = np.std(self.high_noise_region[:, 1])
        self.high_noise_sig = np.mean(self.high_noise_region[:, 1])
        self.low_noise1 = np.std(self.low_noise_region1[:, 1])
        self.low_noise2 = np.std(self.low_noise_region2[:, 1])
        self.low_noise_std = np.std(self.low_noise_region2[:, 1])
        self.low_noise_sig = np.mean(self.low_noise_region2[:, 1])
        self.low_noise3 = np.std(self.low_noise_region3[:, 1])

        # self.noise_list = [self.high_noise, self.low_noise1, self.low_noise2, self.low_noise3]

        self.peak1 = np.array(self.ccd_data[303:323, :])
        self.peak2 = np.array(self.ccd_data[319:339, :])
        self.peak3 = np.array(self.ccd_data[736:746, :])
        self.peak4 = np.array(self.ccd_data[1268:1288, :])
        self.peak5 = np.array(self.ccd_data[1381:1421, :])

        temp_max = np.argmax(self.peak1[:, 1])
        self.signal1 = np.sum(self.peak1[temp_max - 1:temp_max + 2, 1])
        self.error1 = np.sqrt(np.sum(self.peak1[temp_max - 1:temp_max + 2, 2] ** 2))

        temp_max = np.argmax(self.peak2[:, 1])
        self.signal2 = np.sum(self.peak2[temp_max - 1:temp_max + 2, 1])
        self.error2 = np.sqrt(np.sum(self.peak2[temp_max - 1:temp_max + 2, 2] ** 2))

        temp_max = np.argmax(self.peak3[:, 1])
        self.signal3 = np.sum(self.peak3[temp_max - 1:temp_max + 2, 1])
        self.error3 = np.sqrt(np.sum(self.peak3[temp_max - 1:temp_max + 2, 2] ** 2))

        temp_max = np.argmax(self.peak4[:, 1])
        self.signal4 = np.sum(self.peak4[temp_max - 1:temp_max + 2, 1])
        self.error4 = np.sqrt(np.sum(self.peak4[temp_max - 1:temp_max + 2, 2] ** 2))

        temp_max = np.argmax(self.peak5[:, 1])
        self.signal5 = np.sum(self.peak5[temp_max - 1:temp_max + 2, 1])
        self.error5 = np.sqrt(np.sum(self.peak5[temp_max - 1:temp_max + 2, 2] ** 2))

        self.signal_list = [self.signal1, self.signal2, self.signal3, self.signal4, self.signal5]
        self.error_list = [self.error1, self.error2, self.error3, self.error4, self.error5]
        print("Signal list:", self.signal_list)
        self.ccd_data = np.flipud(self.ccd_data)

    def process_stuff(self):
        """
        This one puts high_noise, low_noise1, signal2, and error2 in a nice horizontal array
        """
        # self.results = np.array([self.high_noise, self.low_noise1, self.signal5, self.error5])
        # average = np.mean([self.low_noise1, self.low_noise2, self.low_noise3])
        # self.results = np.array([self.high_noise, self.low_noise1, self.low_noise2, self.low_noise3, self.high_noise/average])
        self.results = np.array([self.high_noise_sig, self.high_noise_std, self.low_noise_sig, self.low_noise_std])

def collect_noise(neon_list, param_name, folder_name, file_name, name='Signal'):
    """
    This function acts like save parameter sweep.

    param_name = string that we're gonna save!
    """
    # param_array = None
    for elem in neon_list:
        print("pname: {}".format(elem.parameters[param_name]))
        print("results:", elem.results)
        temp = np.insert(elem.results, 0, elem.parameters[param_name])
        try:
            param_array = np.row_stack((param_array, temp))
        except UnboundLocalError:
            param_array = np.array(temp)

    if len(param_array.shape) == 1:
        print("I don't think you want this file")
        return
        # append the relative peak error

    print('\n', param_array, '\n')

    param_array = np.column_stack((param_array, param_array[:, 4] / param_array[:, 3]))
    # append the snr
    param_array = np.column_stack((param_array, param_array[:, 3] / param_array[:, 2]))

    try:
        param_array = param_array[param_array[:, 0].argsort()]
    except:
        print("param_array shape", param_array.shape)
        raise

    try:
        os.mkdir(folder_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    file_name = file_name + '.txt'

    origin_import1 = param_name + ",Noise,Noise,Signal,error,rel peak error,peak signal-to-noise"
    # origin_import1 = param_name + ",Noise,Noise,Noise,Noise,Ratio"
    origin_import2 = ",counts,counts,counts,counts,,"
    # origin_import2 = ",counts,counts,counts,,"
    origin_import3 = ",High noise region,Low noise region,{},{} error,{} rel error, {}".format(name, name, name, name)
    # origin_import3 = ",High noise region,Low noise region 1,Low noise region 2,Low noise region 3,High/low"
    header_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    # print "Spec header: ", spec_header
    print("the param_array is:", param_array)
    np.savetxt(os.path.join(folder_name, file_name), param_array, delimiter=',',
               header=header_total, comments='', fmt='%0.6e')
    print("Saved the file.\nDirectory: {}".format(os.path.join(folder_name, file_name)))
