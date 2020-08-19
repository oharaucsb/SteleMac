import os
import io
import glob
import errno
import copy
import json
import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

np.set_printoptions(linewidth=500)


def fvb_crr(raw_array, offset=0, medianRatio=1, noiseCoeff=5, debugging=False):
    """

        Remove cosmic rays from a sequency of identical exposures
        :param raw_array: The array to be cleaned. Successive spectra should
                be the columns (i.e. 1600 x n) of the raw_array
        :param offset: baseline to add to raw_array.
               Not used, but here if it's needed in the future
        :param medianRatio: Multiplier to the median when deciding a cutoff
        :param noiseCoeff: Multiplier to the noise on the median
                    May need changing for noisy data
        :return:
    """

    d = np.array(raw_array)

    med = ndimage.filters.median_filter(d, size=(1, d.shape[1]), mode='wrap')
    med = np.median(d, axis=1).reshape(d.shape[0], 1)
    if debugging:
        print("shape of median filter:", med.shape)
    meanMedian = med.mean(axis=1)
    # meanMedian = med.copy()
    if debugging:
        print("shape of meaned median filter:", meanMedian.shape)
    # Construct a cutoff for each pixel. It was kind of guess and
    # check
    cutoff = meanMedian * medianRatio + noiseCoeff * np.std(meanMedian[-100:])
    if debugging:
        print("shape of cutoff criteria:", cutoff.shape)
        import pyqtgraph as pg

        winlist = []
        app = pg.QtGui.QApplication([])

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("Raw Image")
        p1 = win.addPlot()

        img = pg.ImageItem()
        img.setImage(d.copy().T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win.addItem(hist)

        win.nextRow()
        p2 = win.addPlot(colspan=2)
        p2.setMaximumHeight(250)
        p2.addLegend()
        for i, v in enumerate(d.T):
            p2.plot(v, pen=(i, d.shape[1]), name=str(i))
        p2.plot(np.sum(d, axis=1), pen=pg.mkPen('w', width=3))
        win.show()
        winlist.append(win)

        win2 = pg.GraphicsLayoutWidget()
        win2.setWindowTitle("Median Image")
        p1 = win2.addPlot()

        img = pg.ImageItem()
        img.setImage(med.T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win2.addItem(hist)

        win2.nextRow()
        p2 = win2.addPlot(colspan=2)
        p2.setMaximumHeight(250)

        p2.plot(np.sum(med, axis=1) / d.shape[1])
        win2.show()
        winlist.append(win2)

        win2 = pg.GraphicsLayoutWidget()
        win2.setWindowTitle("d-m")
        p1 = win2.addPlot()

        img = pg.ImageItem()
        img.setImage((d - med).T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win2.addItem(hist)

        win2.nextRow()
        p2 = win2.addPlot(colspan=2)
        p2.setMaximumHeight(250)
        p2.addLegend()
        for i, v in enumerate((d - med).T):
            p2.plot(v, pen=(i, d.shape[1]), name=str(i))
        p2.plot(cutoff, pen=pg.mkPen('w', width=3))
        win2.show()
        winlist.append(win2)

    # Find the bad pixel positions
    # Note the [:, None] - needed to cast the correct shapes
    badPixs = np.argwhere((d - med) > (cutoff.reshape(len(cutoff), 1)))

    for pix in badPixs:
        # get the other pixels in the row which aren't the cosmic
        if debugging:
            print("cleaning pixel", pix)
        p = d[pix[0], [i for i in range(d.shape[1]) if not i == pix[1]]]
        if debugging:
            print("\tRemaining pixels in row are", p)
        # Replace the cosmic by the average of the others
        # Could get hairy if more than one cosmic per row.
        # Maybe when doing many exposures?
        d[pix[0], pix[1]] = np.mean(p)

    if debugging:
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("Clean Image")
        p1 = win.addPlot()

        img = pg.ImageItem()
        img.setImage(d.copy().T)
        p1.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win.addItem(hist)

        win.nextRow()
        p2 = win.addPlot(colspan=2)
        p2.setMaximumHeight(250)
        p2.plot(np.sum(d, axis=1))
        win.show()
        winlist.append(win)
        app.exec_()

    return np.array(d)

def stitchData(dataList, plot=False):
    """
    Attempt to stitch together absorbance data. Will translate the second data set
    to minimize leastsq between the two data sets.
    :param dataList: Iterable of the data sets to be fit. Currently
            it only takes the first two elements of the list, but should be fairly
            straightforward to recursivly handle a list>2. Shifts the second
            data set to overlap the first

             elements of dataList can be either np.arrays or Absorbance class,
              where it will take the proc_data itself
    :param plot: bool whether or not you want the fit iterations to be plotted
            (for debugging)
    :return: a, a (2,) np.array of the shift
    """

    # Data coercsion, make sure we know what we're working wtih
    first = dataList[0]
    if isinstance(first, Absorbance):
        first = first.proc_data
        second = dataList[1]
    if isinstance(second, Absorbance):
        second = second.proc_data
    if plot:
        # Keep a reference to whatever plot is open at call-time
        # Useful if the calling script has plots before and after, as
        # omitting this will cause future plots to be added to figures here
        firstFig = plt.gcf()
        plt.figure("Stitcher")
        # Plot the raw input data
        plt.plot(*first.T)
        plt.plot(*second.T)

    # Algorithm is set up such that the "second" data set spans the
    # higher domain than first. Need to enforce this, and remember it
    # so the correct shift is applied
    flipped = False
    if max(first[:, 0]) > max(second[:, 0]):
        flipped = True
        first, second = second, first

def save_parameter_sweep_no_sb(spectrum_list, file_name, folder_str, param_name, unit,
                         verbose=False):
    """
    This function will take a fully processed list of spectrum objects and
    slice Spectrum.sb_fits appropriately to get an output like:

    "Parameter" | SB1 freq | err | SB1 amp | error | SB1 linewidth | error | SB2...| SBn...|
    param1      |    .     |
    param2      |    .     |
      .
      .
      .

    Currently I'm thinking fuck the offset y0
    After constructing this large matrix, it will save it somewhere.
    """
    spectrum_list.sort(key=lambda x: x.parameters[param_name])
    included_spectra = dict()
    param_array = None
    sb_included = []

    for spec in spectrum_list:
        sb_included = sorted(list(set(sb_included + list(spec.full_dict.keys()))))
        included_spectra[spec.fname.split('/')[-1]] = spec.parameters[param_name]
        # If these are from summed spectra, then only the the first file name
        # from that sum will show up here, which should be fine?
    if verbose:
        # print "full name:", spectrum_list[0].fname
        print("included names:", included_spectra)
        print("sb_included:", sb_included)

    for spec in spectrum_list:
        temp_dict = {}  # This is different from full_dict in that the list has the
        # sideband order as the zeroth element.
        if verbose:
            print("the sb_results:", spec.sb_results)
        if spec.sb_results.ndim == 1: continue
        for index in range(len(spec.sb_results[:, 0])):
            if verbose:
                print("my array slice:", spec.sb_results[index, :])
            temp_dict[int(round(spec.sb_results[index, 0]))] = np.array(
                spec.sb_results[index, 1:])

        if verbose:
            print(temp_dict)

        for sb in sb_included:
            blank = np.zeros(6)
            # print "checking sideband order:", sb
            # print "blank", blank
            if sb not in temp_dict:
                # print "\nNeed to add sideband order:", sb
                temp_dict[sb] = blank
        try:  # Why is this try-except here?
            spec_data = np.array([float(spec.parameters[param_name])])
        except:
            spec_data = np.array([float(spec.parameters[param_name][:2])])
        for key in sorted(temp_dict.keys()):
            # print "I am going to hstack this:", temp_dict[key]
            spec_data = np.hstack((spec_data, temp_dict[key]))

        try:
            param_array = np.vstack((param_array, spec_data))
        except:
            param_array = np.array(spec_data)
        if verbose:
            print("The shape of the param_array is:", param_array.shape)
            # print "The param_array itself is:", param_array
    '''
    param_array_norm = np.array(param_array).T # python iterates over rows
    for elem in [x for x in xrange(len(param_array_norm)) if (x-1)%7 == 3]:
        temp_max = np.max(param_array_norm[elem])
        param_array_norm[elem] = param_array_norm[elem] / temp_max
        param_array_norm[elem + 1] = param_array_norm[elem + 1] / temp_max
    '''
    snipped_array = param_array[:, 0]
    norm_array = param_array[:, 0]
    if verbose:
        print("Snipped_array is", snipped_array)
    for ii in range(len(param_array.T)):
        if (ii - 1) % 6 == 0:
            if verbose:
                print("param_array shape", param_array[:, ii])
            snipped_array = np.vstack((snipped_array, param_array[:, ii]))
            norm_array = np.vstack((norm_array, param_array[:, ii]))
        elif (ii - 1) % 6 == 2:
            snipped_array = np.vstack((snipped_array, param_array[:, ii]))

            temp_max = np.max(param_array[:, ii])
            norm_array = np.vstack((norm_array, param_array[:, ii] / temp_max))
        elif (ii - 1) % 6 == 3:
            snipped_array = np.vstack((snipped_array, param_array[:, ii]))
            norm_array = np.vstack((norm_array, param_array[:, ii] / temp_max))

    snipped_array = snipped_array.T
    norm_array = norm_array.T

    try:
        os.mkdir(folder_str)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    norm_name = file_name + '_norm.txt'
    snip_name = file_name + '_snip.txt'
    file_name = file_name + '.txt'

    try:
        included_spectra_str = json.dumps(included_spectra, sort_keys=True, indent=4,
                                          separators=(',', ': '))
    except:
        print("Source: save_parameter_sweep\nJSON FAILED")
        return
    included_spectra_str = included_spectra_str.replace('\n', '\n#')

    included_spectra_str += '\n#' * (99 - included_spectra_str.count('\n'))
    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    for order in sb_included:
        origin_import1 += "Frequency,error,Sideband strength,error,Linewidth,error"
        origin_import2 += ",eV,,arb. u.,,meV,"
        origin_import3 += ",{0},,{0},,{0},".format(order)
    origin_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    for order in sb_included:
        origin_import1 += ",Frequency,Sideband strength,error"
        origin_import2 += ",eV,arb. u.,"
        origin_import3 += ",{0},{0},".format(order)
    origin_snip = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    header_total = '#' + included_spectra_str + '\n' + origin_total
    header_snip = '#' + included_spectra_str + '\n' + origin_snip

    # print "Spec header: ", spec_header
    if verbose:
        print("the param_array is:", param_array)
    np.savetxt(os.path.join(folder_str, file_name), param_array, delimiter=',',
               header=header_total, comments='', fmt='%0.6e')
    np.savetxt(os.path.join(folder_str, snip_name), snipped_array, delimiter=',',
               header=header_snip, comments='', fmt='%0.6e')
    np.savetxt(os.path.join(folder_str, norm_name), norm_array, delimiter=',',
               header=header_snip, comments='', fmt='%0.6e')
    if verbose:
        print("Saved the file.\nDirectory: {}".format(
            os.path.join(folder_str, file_name)))

def save_parameter_sweep(spectrum_list, file_name, folder_str, param_name, unit,
                         wanted_indices = [1, 3, 4], skip_empties = False, verbose=False,
                         header_dict = {}, only_even=False):
    """
    This function will take a fully processed list of spectrum objects and
    slice Spectrum.sb_fits appropriately to get an output like:

    "Parameter" | SB1 freq | err | SB1 amp | error | SB1 linewidth | error | SB2...| SBn...|
    param1      |    .     |
    param2      |    .     |
      .
      .
      .

    Currently I'm thinking fuck the offset y0
    After constructing this large matrix, it will save it somewhere.


    Thus function has been update to pass a list of indices to slice for the return
    values

    skip_empties: If False, will add a row of zeroes for the parameter even if no sidebands
    are found. If True, will not add a line for that parameter

    only_even: don't include odd orders in the saved sweep

    [sb number, Freq (eV), Freq error (eV), Gauss area (arb.), Area error, Gauss linewidth (eV), Linewidth error (eV)]
    [    0    ,      1   ,        2,      ,        3         ,      4    ,         5           ,        6            ]
    """
    if isinstance(param_name, list):
        # if you pass two things because the param you want
        # is in a dict (e.g. field strength has mean/std)
        # do it that way
        param_name_list = list(param_name) # keep reference to old one
        paramGetter = lambda x: x.parameters[param_name_list[0]][param_name_list[1]]
        # Keep the name for labeling things later on
        param_name = param_name[0]
    else:
        paramGetter = lambda x: x.parameters[param_name]

    # Sort all of the spectra based on the desired key
    spectrum_list.sort(key=paramGetter)

    # keep track of which file name corresponds to which parameter which gets put in
    included_spectra = dict()

    # The big array which will be stacked up to keep all of the sideband details vs desired parameter
    param_array = None

    # list of which sidebands are seen throughout.
    sb_included = []
    # how many parameters (area, strength, linewidth, pos, etc.) are there?
    # Here incase software changes and more things are kept in
    # sb results. Needed to handle how to slice the arrays
    try:
        num_params = spectrum_list[0].sb_results.shape[1]
    except IndexError:
        # There's a file with only 1 sb and it happens to be first
        # in the list.
        num_params = spectrum_list[0].sb_results.shape[0]
    except AttributeError:
        # The first file has no sidebands, so just hardcode it, as stated below.
        num_params=0

    # Rarely, there's an issue where I'm doing some testing and there's a set
    # where the first file has no sidebands in it, so the above thing returns 0
    # It seems really silly to do a bunch of testing to try and correct for that, so
    # I'm going to hardcode the number of parameters.
    if num_params == 0:
        num_params = 7

    # loop through all of them once to figure out which sidebands are seen in all spectra
    for spec in spectrum_list:
        try:
            # use sets to keep track of only unique sidebands
            sb_included = sorted(list(set(sb_included + list(spec.full_dict.keys()))))
        except AttributeError:
            print("No full dict?", spec.fname)
            print(spec.sb_list)
        # If these are from summed spectra, then only the the first file name
        # from that sum will show up here, which should be fine?
        included_spectra[spec.fname.split('/')[-1]] = paramGetter(spec)

    if only_even:
        sb_included = [ii for ii in sb_included if not ii%2]
    if verbose:
        print("included names:", included_spectra)
        print("sb_included:", sb_included)

    for spec in spectrum_list:
        # Flag to keep whethere there are no sidebands or not. Used to skip
        # issues when trying to index on empty arrays
        noSidebands = False
        if verbose:
            print("the sb_results:", spec.sb_results)

        # if no sidebands were found, skip this one
        try:
            # TODO: (08/14/18) the .ndim==1 isn't the correct check, since it fails
            # when looking at the laser line. Need to test this with a real
            # empty data set, vs data set with 1 sb
            #
            #
            # (08/28/18) I'm not sure what the "not spec" is trying to handle
            #      spec.sb_results is None occurs when _no_ sidebands were fit
            #     spec.sb_results.ndim == 1 happens when only one sideband is found
            if not spec or spec.sb_results is None or spec.sb_results.ndim == 1:
                if spec.sb_results is None:
                    # Flag no sidebands are afound
                    noSidebands = True
                elif spec.sb_results[0] == 0:
                    # Cast it to 2d to allow slicing later on. Not sure hwy this is
                    # only done if the laser line is the one found.
                    spec.sb_results = np.atleast_2d(spec.sb_results)
                elif skip_empties:
                    continue
                else:
                    noSidebands = True
        except (AttributeError, TypeError):
            # continue
            raise

        # Make an sb_results of all zeroes where we'll fill
        # in the sideband info we found
        new_spec = np.zeros((len(sb_included), num_params))
        if not noSidebands:
            sb_results = spec.sb_results.copy()
            saw_sbs = sb_results[:, 0]
            found_sb = sorted(list(set(sb_included) & set(saw_sbs)))
            found_idx = [sb_included.index(ii) for ii in found_sb]
            try:
                new_spec[:, 0] = sb_included
            except:
                print("new_spec", new_spec)
                raise
            try:
                if only_even:
                    new_spec[found_idx, :] = sb_results[sb_results[:,0]%2==0]
                else:
                    new_spec[found_idx, :] = sb_results
            except ValueError:
                print(spec.fname)
                print("included:", sb_included)
                print("found:", found_sb, found_idx)
                print(new_spec.shape, sb_results.shape)
                print(sb_results)
                print(new_spec)
                raise

        spec_data = np.insert(new_spec.flatten(), 0, float(paramGetter(spec)))

        try:
            param_array = np.row_stack((param_array, spec_data))
        except:
            param_array = np.array(spec_data)

    if param_array.ndim == 1: # if you only pass one spectra
        param_array = param_array[None, :] # recast it to 2D for slicing
    # the indices we want from the param array from the passed argument
    snip = wanted_indices
    N = len(sb_included)
    # run it out across all of the points across the param_array
    snipped_indices = [0] + list(
        1+np.array(snip * N) + num_params * np.array(sorted(list(range(N)) * len(snip))))
    snipped_array = param_array[:, snipped_indices]
    norm_array = snipped_array.copy()
    # normalize the area if it's requested
    if 3 in snip:
        num_snip = len(snip)
        strength_idx = snip.index(3)
        if 4 in snip:
            #normalize error first if it was requested
            idx = snip.index(4)
            norm_array[:, 1 + idx + np.arange(N) * num_snip] /= norm_array[:,1 + strength_idx + np.arange(N) * num_snip].max(axis=0)
        strength_idx = snip.index(3)
        norm_array[:, 1+strength_idx+np.arange(N)*num_snip]/=norm_array[:, 1+strength_idx+np.arange(N)*num_snip].max(axis=0)

    try:
        os.mkdir(folder_str)
    except TypeError:
        pass # if you pass None as folder_str (for using byteIO)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    included_spectra.update(header_dict)
    try:
        included_spectra_str = json.dumps(included_spectra, sort_keys=True, indent=4,
                                          separators=(',', ': '))
    except:
        print("Source: save_parameter_sweep\nJSON FAILED")
        return
    included_spectra_str = included_spectra_str.replace('\n', '\n#')

    included_spectra_str += '\n#' * (99 - included_spectra_str.count('\n'))

    # this will make the header chunk for the full, un-sliced data set
    # TODO: fix naming so you aren't looping twice
    ### 1/9/18 This isn't needed, right? Why isn't it deleted?
    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    for order in sb_included:
        origin_import1 += ",sideband,Frequency,error,Sideband strength,error,Linewidth,error"
        origin_import2 += ",order,eV,eV,arb. u.,arb.u.,meV,meV"
        origin_import3 += ",,{0},,{0},,{0},".format(order)
    origin_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3


    # This little chunk will make a chunk block of header strings for the sliced
    # data set which can be looped over
    origin_import1 = param_name
    origin_import2 = unit
    origin_import3 = ""
    wanted_titles = ["Sideband", "Frequency", "error", "Sideband strength","error","Linewidth","error"]
    wanted_units  = ["order", "eV", "eV", "arb. u.", "arb. u.", "eV", "eV"]
    wanted_comments = ["", "{0}", "", "{0}", "", "{0}", ""]
    wanted_titles = ",".join([wanted_titles[ii] for ii in wanted_indices])
    wanted_units = ",".join([wanted_units[ii] for ii in wanted_indices])
    wanted_comments = ",".join([wanted_comments[ii] for ii in wanted_indices])

    for order in sb_included:
        origin_import1 += ","+wanted_titles
        origin_import2 += ","+wanted_units
        origin_import3 += ","+wanted_comments.format(order)
    origin_snip = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    header_total = '#' + included_spectra_str + '\n' + origin_total
    header_snip = '#' + included_spectra_str + '\n' + origin_snip

    # print "Spec header: ", spec_header
    if verbose:
        print("the param_array is:", param_array)
    if isinstance(file_name, list):
        if isinstance(file_name[0], io.BytesIO):
            np.savetxt(file_name[0], param_array, delimiter=',',
                       header=header_total, comments='', fmt='%0.6e')
            np.savetxt(file_name[1], snipped_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            np.savetxt(file_name[2], norm_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            # Need to reset the file position if you want to read them immediately
            # Is it better to do that here, or assume you'll do it later?
            # I'm gonna assume here, because I can't currently think of a time when I'd want
            # to be at the end of the file
            [ii.seek(0) for ii in file_name]
            if verbose:
                print("Saved the file to bytes objects")
    else:
        if file_name:
            norm_name = file_name + '_norm.txt'
            snip_name = file_name + '_snip.txt'
            file_name = file_name + '.txt'
            np.savetxt(os.path.join(folder_str, file_name), param_array, delimiter=',',
                       header=header_total, comments='', fmt='%0.6e')
            np.savetxt(os.path.join(folder_str, snip_name), snipped_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            np.savetxt(os.path.join(folder_str, norm_name), norm_array, delimiter=',',
                       header=header_snip, comments='', fmt='%0.6e')
            if verbose:
                print("Saved the file.\nDirectory: {}".format(os.path.join(folder_str, file_name)))
        else:
            if verbose:
                print("Didn't save")

    return sb_included, param_array, snipped_array, norm_array

def save_parameter_sweep_vs_sideband(spectrum_list, file_name,
                                     folder_str, param_name, unit, verbose=False,
                                     wanted_indices = [1, 3, 4]):
    """
    Similar to save_parameter_sweep, but the data[:,0] column is sideband number instead of
    series, and each set of columns correspond to a series step. Pretty much compiles
    all of the fit parameters from the files that are already saved and puts it into
    one file to keep from polluting the Origin folder
    :param spectrum_list:
    :param file_name:
    :param folder_str:
    :param param_name:
    :param unit:
    :param verbose:

    sb number is automatically prepended, so do not include in slicing list

    [sb number, Freq (eV), Freq error (eV), Gauss area (arb.), Area error, Gauss linewidth (eV), Linewidth error (eV)]
    [    0    ,      1   ,        2,      ,        3         ,      4    ,         5           ,        6            ]

    :return:
    """
    spectrum_list.sort(key=lambda x: x.parameters[param_name])
    included_spectra = dict()
    param_array = None
    sb_included = []

    # what parameters were included (for headers)
    params = sorted([x.parameters[param_name] for x in spectrum_list])

    for spec in spectrum_list:
        sb_included = sorted(list(set(sb_included + list(spec.full_dict.keys()))))
        included_spectra[spec.fname.split('/')[-1]] = spec.parameters[param_name]
        # If these are from summed spectra, then only the the first file name
        # from that sum will show up here, which should be fine?
    if verbose:
        # print "full name:", spectrum_list[0].fname
        print("included names:", included_spectra)
        print("sb_included:", sb_included)

    param_array = np.array(sb_included)

    for spec in spectrum_list:
        temp_dict = spec.full_dict.copy()

        #prevent breaking if no sidebands in spectrum
        if not temp_dict:
            if verbose:
                print("No sidebands here? {}, {}".format(spec.parameters["series"],
                                                         spec.parameters["spec_step"]))
            continue

        if verbose:
            print(temp_dict)

        # matrix for holding all of the sb information
        # for a given spectrum
        spec_matrix = None
        for sb in sb_included:
            blank = np.zeros(6)
            # print "checking sideband order:", sb
            # print "blank", blank
            sb_data = temp_dict.get(sb, blank)
            try:
                spec_matrix = np.row_stack((spec_matrix, sb_data))
            except:
                spec_matrix = sb_data
        param_array = np.column_stack((param_array, spec_matrix))

    # the indices we want from the param array
    # 1- freq, 3-area, 4-area error
    snip = wanted_indices
    N = len(spectrum_list)
    # run it out across all of the points across the param_array
    snipped_indices = [0] + list( np.array(snip*N) + 6*np.array(sorted(list(range(N))*len(snip))) )
    snipped_array = param_array[:, snipped_indices]

    try:
        os.mkdir(folder_str)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    snip_name = file_name + '_snip.txt'
    file_name = file_name + '.txt'

    try:
        included_spectra_str = json.dumps(included_spectra, sort_keys=True, indent=4, separators=(',', ': '))
    except:
        print("Source: save_parameter_sweep\nJSON FAILED")
        return
    included_spectra_str = included_spectra_str.replace('\n', '\n#')

    included_spectra_str += '\n#' * (99 - included_spectra_str.count('\n'))
    origin_import1 = "Sideband"
    origin_import2 = "Order"
    origin_import3 = "SB"
    for param in params:
        origin_import1 += ",Frequency,error,Sideband strength,error,Linewidth,error"
        origin_import2 += ",eV,,arb. u.,,meV,"
        origin_import3 += ",{0},,{0},,{0},".format(param)
    origin_total = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    # This little chunk will make a chunk block of header strings for the sliced
    # data set which can be looped over
    origin_import1 = "Sideband"
    origin_import2 = "Order"
    origin_import3 = "SB"
    wanted_titles = ["Sideband", "Frequency", "error", "Sideband strength", "error",
                     "Linewidth", "error"]
    wanted_units = ["order", "eV", "eV", "arb. u.", "arb. u.", "eV", "eV"]
    wanted_comments = ["", "{0}", "", "{0}", "", "{0}", ""]
    wanted_titles = ",".join([wanted_titles[ii] for ii in wanted_indices])
    wanted_units = ",".join([wanted_units[ii] for ii in wanted_indices])
    wanted_comments = ",".join([wanted_comments[ii] for ii in wanted_indices])

    for param in params:
        origin_import1 += "," + wanted_titles
        origin_import2 += "," + wanted_units
        origin_import3 += "," + wanted_comments.format(param)
    origin_snip = origin_import1 + "\n" + origin_import2 + "\n" + origin_import3

    header_total = '#' + included_spectra_str + '\n' + origin_total
    header_snip = '#' + included_spectra_str + '\n' + origin_snip

    # print "Spec header: ", spec_header
    if verbose:
        print("the param_array is:", param_array)
    if file_name: # allow passing false (or empty string) to prevent saving
        np.savetxt(os.path.join(folder_str, file_name), param_array, delimiter=',',
                   header=header_total, comments='', fmt='%0.6e')
        np.savetxt(os.path.join(folder_str, snip_name), snipped_array, delimiter=',',
                   header=header_snip, comments='', fmt='%0.6e')
    if verbose:
        print("Saved the file.\nDirectory: {}".format(os.path.join(folder_str, file_name)))
    return None

def stitchData(dataList, plot=False):
    """
    Attempt to stitch together absorbance data. Will translate the second data set
    to minimize leastsq between the two data sets.
    :param dataList: Iterable of the data sets to be fit. Currently
            it only takes the first two elements of the list, but should be fairly
            straightforward to recursivly handle a list>2. Shifts the second
            data set to overlap the first
             elements of dataList can be either np.arrays or Absorbance class,
              where it will take the proc_data itself
    :param plot: bool whether or not you want the fit iterations to be plotted
            (for debugging)
    :return: a, a (2,) np.array of the shift
    """

    # Data coercsion, make sure we know what we're working wtih
    first = dataList[0]
    if isinstance(first, Absorbance):
        first = first.proc_data
    second = dataList[1]
    if isinstance(second, Absorbance):
        second = second.proc_data
    if plot:
        # Keep a reference to whatever plot is open at call-time
        # Useful if the calling script has plots before and after, as
        # omitting this will cause future plots to be added to figures here
        firstFig = plt.gcf()
        plt.figure("Stitcher")
        # Plot the raw input data
        plt.plot(*first.T)
        plt.plot(*second.T)

    # Algorithm is set up such that the "second" data set spans the
    # higher domain than first. Need to enforce this, and remember it
    # so the correct shift is applied
    flipped = False
    if max(first[:, 0]) > max(second[:, 0]):
        flipped = True
        first, second = second, first

    def fitter(p, shiftable, immutable):
        # designed to over

        # Get the shifts
        dx = p[0]
        dy = p[1]

        # Don't want pass-by-reference nonsense, recast our own refs
        shiftable = np.array(shiftable)
        immutable = np.array(immutable)

        # Shift the data set
        shiftable[:, 1] += dy
        shiftable[:, 0] += dx

        # Create an interpolator. We want a
        # direct comparision for subtracting the two functions
        # Different spec grating positions have different wavelengths
        # so they're not directly comparable.
        shiftF = spi.interp1d(*shiftable.T)

        # Find the bounds of where the two data sets overlap
        overlap = (min(shiftable[:, 0]), max(immutable[:, 0]))
        print("overlap", overlap)

        # Determine the indices of the immutable function
        # where it overlaps. argwhere returns 2-d thing,
        # requiring the [0] at the end of each call
        fOlIdx = (min(np.argwhere(immutable[:, 0] >= overlap[0]))[0],
                  max(np.argwhere(immutable[:, 0] <= overlap[1]))[0])
        print("fOlIdx", fOlIdx)

        # Get the interpolated values of the shiftable function at the same
        # x-coordinates as the immutable case
        newShift = shiftF(immutable[fOlIdx[0]:fOlIdx[1], 0])

        if plot:
            plt.plot(*immutable[fOlIdx[0]:fOlIdx[1], :].T, marker='o', label="imm", markersize=10)
            plt.plot(immutable[fOlIdx[0]:fOlIdx[1], 0], newShift, marker='o', label="shift")
        imm = immutable[fOlIdx[0]:fOlIdx[1], 1]
        shift = newShift
        return imm - shift

    a, _, _, msg, err = spo.leastsq(fitter, [0.0001, 0.01 * max(first[:, 1])], args=(second, first), full_output=1)
    # print "a", a
    if plot:
        # Revert back to the original figure, as per top comments
        plt.figure(firstFig.number)

    # Need to invert the shift if we flipped which
    # model we're supposed to move
    if flipped: a *= -1

    return a


def integrateData(data, t1, t2, ave=False):
    """
    Integrate a discrete data set for a
    given time period. Sums the data between
    the given bounds and divides by dt. Optional
    argument to divide by T = t2-t1 for calculating
    averages.

    data = 2D array. data[:,0] = t, data[:,1] = y
    t1 = start of integration
    t2 = end of integration


    if data is a NxM, with M>=3, it will take the
    third column to be the errors of the points,
    and return the error as the quadrature sum
    """
    t = data[:, 0]
    y = data[:, 1]
    if data.shape[0] >= 3:
        errors = data[:, 2]
    else:
        errors = np.ones_like(y) * np.nan

    gt = set(np.where(t > t1)[0])
    lt = set(np.where(t < t2)[0])

    # find the intersection of the sets
    vals = list(gt & lt)

    # Calculate the average
    tot = np.sum(y[vals])
    error = np.sqrt(np.sum(errors[vals] ** 2))

    # Multiply by sampling
    tot *= (t[1] - t[0])
    error *= (t[1] - t[0])

    if ave:
        # Normalize by total width if you want an average
        tot /= (t2 - t1)
        errors /= (t2 - t1)
    if not np.isnan(error):
        return tot, error
    return tot


def get_data_and_header(fname, returnOrigin=False):
    """
    Given a file to a raw data file, returns the data
    and the json decoded header.

    Can choose to return the origin header as well
    :param fname: Filename to open
    :return: data, header (dict)
    """
    with open(fname) as fh:
        line = fh.readline()
        header_string = ''
        while line[0]=='#':
            header_string += line[1:]
            line = fh.readline()

        # image files don't have an origin header
        if not "Images" in fname:
            oh = line
            # last readline in loop removes first line in Origin Header
            # strip the remaining two
            oh += fh.readline()
            oh += fh.readline()[:-1] #remove final \n

        # data = np.genfromtxt(fh, delimiter=',')
    data = np.genfromtxt(fname, delimiter=',')

    header = json.loads(header_string)

    if returnOrigin:
        return data, header, oh
    return data, header

def natural_glob(*args):
    # glob/python sort alphabetically, so 1, 10, 11, .., 2, 21,
    # but I sometimes wnat "natural" sorting: 1, 2, 3, ..., 10, 11, 12, ..., 20, 21, 21 ...
    # There's tons of stack overflows, so I grabbed one of them. I put it in here
    # because I use it all the damned time. I also almost always use it when
    # glob.glob'ing, so just internally do it that way
    #
    # This is taken from
    # https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    import re
    def atoi(text):
        try:
            return int(text)
        except ValueError:
            return text
        # return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split('(-?\d+)', text)]

    return sorted(glob.glob(os.path.join(*args)), key=natural_keys)

def convertTime(timeStr):
    """
    The data file headers have the timestamp of data collection. Sometimes you want to
    convert that to numbers for data's sake, but I constantly forget the functions
    to convert it from the time-stamp string. So here you go
    :param timeStr: the time as a string from the data file
    :return: int of the time since the epoch
    """
    import time
    return time.mktime(time.strptime(timeStr, "%x %X%p"))
