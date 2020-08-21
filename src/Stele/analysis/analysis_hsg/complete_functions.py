import os
import glob
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=500)


def proc_n_plotPMT(folder_path, plot=False, confirm_fits=False, save=None, verbose=False, **kwargs):
    """
    This function will take a pmt object, process it completely.

    :rtype: list of HighSidebandPMT
    """
    pmt_data = pmt_sorter(folder_path, plot_individual=plot)

    index = 0
    for spectrum in pmt_data:
        spectrum.integrate_sidebands(verbose=verbose, **kwargs)
        spectrum.laser_line(verbose=verbose, **kwargs)  # This function is broken
        # because process sidebands can't handle the laser line
        # print spectrum.full_dict
        if plot:
            plt.figure('PMT data')
            for sb, elem in list(spectrum.sb_dict.items()):
                plt.errorbar(elem[:, 0], elem[:, 1], elem[:, 2],
                             marker='o', label="{} {}".format(spectrum.parameters["series"],sb))
            plt.figure('Sideband strengths')
            plt.yscale("log")
            plt.errorbar(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3], spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
        if plot and confirm_fits:
            plt.figure('PMT confirm fits')
            for elem in list(spectrum.sb_dict.values()):
                plt.errorbar(elem[:, 0], elem[:, 1], elem[:, 2], marker='o')
            plt.errorbar(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3], spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
            plt.ylim([-0.005, 0.025])
        if type(save) is tuple:
            spectrum.save_processing(save[0], save[1], index=index)
            index += 1
        elif isinstance(save, str):
            dirr = os.path.dirname(save) if os.path.dirname(save) else '.' # if you just pass a filename tos ave
            spectrum.save_processing(os.path.basename(save), dirr,
                                     index=index)
            index += 1
    if plot:
        plt.legend()
    return pmt_data


def proc_n_plotCCD(folder_path, offset=None, plot=False, confirm_fits=False,
                   save=None, keep_empties = False, verbose=False, **kwargs):
    """
    This function will take a list of ccd files and process it completely.
    save_name is a tuple (file_base, folder_path)

    keep_empties: If True, keep the HighSidebandCCD object in the list if no sidebands
    are found. Else, cut it off.

    The cutoff of 8 is too high, but I don't know what to change it to
    :rtype: list of HighSidebandCCD
    """
    if isinstance(folder_path, list):
        file_list = folder_path
    else:
        # if verbose:
            # print "Looking in:", os.path.join(folder_path, '*seq_spectrum.txt')
        # file_list = glob.glob(os.path.join(folder_path, '*seq_spectrum.txt'))
        file_list = natural_glob(folder_path, '*seq_spectrum.txt')
        # if verbose:
            # print "found these files:", "\n".join([os.path.basename(ii) for ii in file_list])
    raw_list = []
    for fname in file_list:
        raw_list.append(HighSidebandCCD(fname, spectrometer_offset=offset))

    index = 0
    for spectrum in raw_list:
        try:
            spectrum.guess_sidebands(verbose=verbose, plot=plot)
        except RuntimeError:
            print("\n\n\nNo sidebands??\n\n")
            # No sidebands, say it's empty
            if not keep_empties:
                raw_list.pop(raw_list.index(spectrum))
            continue
        try:
            spectrum.fit_sidebands(plot=plot, verbose=verbose)
        except RuntimeError:
            print("\n\n\nNo sidebands??\n\n")
            # No sidebands, say it's empty
            if not keep_empties:
                raw_list.pop(raw_list.index(spectrum))
            continue
        if "calculated NIR freq (cm-1)" not in list(spectrum.parameters.keys()):
            spectrum.infer_frequencies()
        if plot:
            plt.figure('CCD data')
            plt.errorbar(spectrum.proc_data[:, 0], spectrum.proc_data[:, 1], spectrum.proc_data[:, 2],
                         label=spectrum.parameters['series'])
            plt.legend()
            # plt.yscale('log')
            plt.figure('Sideband strengths')
            plt.errorbar(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3], spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
            plt.legend()
            plt.yscale('log')
        if plot and confirm_fits:
            plt.figure('CCD confirm fits')
            plt.plot(spectrum.proc_data[:, 0], spectrum.proc_data[:, 1],# spectrum.proc_data[:, 2],
                         label=spectrum.parameters['series'])
            plt.plot(spectrum.sb_results[:, 1], spectrum.sb_results[:, 3] / spectrum.sb_results[:, 5],# spectrum.sb_results[:, 4],
                         label=spectrum.parameters['series'], marker='o')
            plt.legend()
            plt.ylim([-0.1, 1])
        if type(save) is tuple:
            spectrum.save_processing(save[0], save[1],
                                     marker=spectrum.parameters["series"].replace(
                                         r"/", "p"),
                                     index=index)
            index += 1
        elif isinstance(save, str):
            # print "DEBUG: trying to save CCD with ", os.path.dirname(save),'_at_', os.path.basename(save)
            spectrum.save_processing(os.path.basename(save), os.path.dirname(save),
                                     marker=spectrum.parameters["series"].replace(
                                         r"/", "p"),
                                     index=index)
            index += 1
    return raw_list


def create_full_spectra(folder_path, skipLaser = True, *args, **kwargs):
    """
    Given the folder path of raw data (where the PMT data is held in the subfolder "PMT"),
    scale all the data to create a raw comb spectra.
    :param folder_path:
    :param args:
    :param kwargs:
    :return:
    """
    output = np.empty((0,2))
    # have proc_n_plot do all the integrating for the sbs
    pmt = proc_n_plotPMT(os.path.join(folder_path, "PMT"))

    ccd_file_list = glob.glob(os.path.join(folder_path, '*seq_spectrum.txt'))
    ccd_list = [HighSidebandCCD(fname) for fname in ccd_file_list]




    for pmtsb in sorted(pmt[0].sb_dict.keys()):
        if skipLaser and pmtsb == 0: continue
        data = pmt[0].sb_dict[pmtsb]
        try:
            print(pmtsb, pmt[0].full_dict[pmtsb])
        except:
            continue
        output = np.row_stack((output, np.abs(data[:,[0,1]])))
        output = np.row_stack((output, [np.nan, np.nan]))

    # insert the pmt so I can iterate over scaling consecutive pairs
    ccd_list.insert(0, pmt[0])

    # make sure all things get scaled down by the factors before them
    runningRatio = 1
    for idx, ccd in enumerate(ccd_list[1:]):
        ccd.guess_sidebands()
        ccd.fit_sidebands()
        ratio = [1, 1]

        # stitch _hsg_dicts moved to Fullspectrum helpers
        stitch_hsg_dicts(ccd_list[idx], ccd, need_ratio = True, ratios=ratio)

        print("new ratio", ratio)
        runningRatio *= ratio[1]
        ccd.proc_data[:,1]*=runningRatio

        output = np.row_stack((output, np.abs(ccd.proc_data[:,[0,1]])))
        output = np.row_stack((output, [np.nan, np.nan]))

    offsetEnergy = (output[:,0] - pmt[0].full_dict[0][0])*1e3
    print(offsetEnergy.shape, output.shape)
    output = np.column_stack((output[:,0], offsetEnergy.T, output[:,1]))

    return output
