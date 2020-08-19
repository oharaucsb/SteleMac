import copy
import numpy as np
from .FullSpectrum_collection import FullHighSideband
from .processing.PMT_collection import HighSidebandPMT as HSPMT


def stitch_hsg_dicts_old(full, new_dict, need_ratio=False, verbose=False):
    """
    This helper function takes a FullHighSideband.full_dict attribute and a
    sideband object, either CCD or PMT and smushes the new sb_results into the
    full_dict.

    The first input doesn't change, so f there's a PMT set of data involved, it
    should be in the full variable to keep the laser normalization intact.

    This function almost certainly does not work for stitching many negative
    orders in it's current state

    11/14/16
    --------
    The original function has been updated to take the full object (instead of
    the dicts alone) to better handle calculating ratios when stitching. This
    is called once things have been parsed in the original function (or legacy
    code where dicts are passed instead of the object)

    Inputs:
    full = full_dict from FullHighSideband, or HighSidebandPMT.  It's important
           that it contains lower orders than the new_dict.
    new_dict = another full_dict.
    need_ratio = If gain or other parameters aren't equal and must resort to
                 calculating the ratio instead of the measurements being
                 equivalent. Changing integration time still means N photons
                 made M counts, but changing gain or using PMT or whatever does
                 affect things.

    Returns:
    full = extended version of the input full.  Overlapping sidebands are
           averaged because that makes sense?
    """
    if verbose:
        print("I'm adding these sidebands in old stitcher", sorted(
            new_dict.keys()))
    # The list that hold which orders are in both dictionaries
    overlap = []
    # How to deal with sidebands that are missing from full but in new.
    missing = []
    for new_sb in sorted(new_dict.keys()):
        full_sbs = sorted(full.keys())
        if new_sb in full_sbs:
            overlap.append(new_sb)
        # This probably doesn't work with bunches of negative orders
        elif new_sb not in full_sbs and new_sb < full_sbs[-1]:
            missing.append(new_sb)

    if verbose:
        print("overlap:", overlap)
        print("missing:", missing)

    # This if-else clause handles how to average together overlapping sidebands
    # which are seen in both spectra,
    if need_ratio:
        # Calculate the appropriate ratio to multiply the new sidebands by.
        # I'm not entirely sure what to do with the error of this guy.
        ratio_list = []
        # print '\n1979\nfull[2]', full[0][2]
        try:
            new_starter = overlap[-1]
            if len(overlap) > 2:
                overlap = [x for x in overlap if (x % 2 == 0)]
                # and (x != min(overlap) and (x != max(overlap)))]
            for sb in overlap:
                ratio_list.append(full[sb][2] / new_dict[sb][2])
            ratio = np.mean(ratio_list)
            # print
            # print '-'*15
            # print "ratio for {}: {}".format()
            error = np.std(ratio_list) / np.sqrt(len(ratio_list))
        except IndexError:
            # If there's no overlap (which you shouldn't let happen),
            # hardcode a ratio and error.
            # I looked at all the ratios for the overlaps from 6/15/16
            # (540ghz para) to get the rough average. Hopefully they hold
            # for all data.
            if not overlap:
                ratio = 0.1695
                error = 0.02
                # no overlap, so make sure it grabs
                # all the sidebands
                new_starter = min(new_dict.keys())
            else:
                raise
        if verbose:
            print("Ratio list", "\n", [round(ii, 3) for ii in ratio_list])
            print("Overlap   ", "\n", [round(ii, 3) for ii in overlap])
            print("Ratio", ratio)
            print("Error", error)
        # print '\n2118\nfull[2]', full[0][2]
        # Adding the new sidebands to the full set and moving errors around.
        # I don't know exactly what to do about the other aspects of the
        # sidebands besides the strength and its error.
        for sb in overlap:
            full[sb][2] = ratio * new_dict[sb][2]
            full[sb][3] = full[sb][2] * np.sqrt(
                (error / ratio) ** 2 +
                (new_dict[sb][3] / new_dict[sb][2]) ** 2)
            # print '\n2125\nfull[2]', full[0][3]
            # Now for linewidths
            lw_error = np.sqrt(
                full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
            lw_avg = (
                    full[sb][4] / (full[sb][5] ** 2) +
                    new_dict[sb][4] / (new_dict[sb][5] ** 2)) / (
                    full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
            full[sb][4] = lw_avg
            full[sb][5] = lw_error
        # print '\n2132\nfull[2]', full[0][2]
    else:
        try:
            # This grabs the sideband order where only the new dictionary has
            # sideband information.  It's not clear why it necessarily has to
            # be at this line.
            new_starter = overlap[-1]
            # This cuts out the lowest order sideband in the overlap for
            # mysterious reasons
            overlap = [
                x for x in overlap if
                (x % 2 == 0) and (x != min(overlap) and (x != max(overlap)))]
            # This for loop average two data points weighted by their
            # relative errors
            for sb in overlap:
                if verbose:
                    print("The sideband", sb)
                    print("Old value", full[sb][4] * 1000)
                    print("Add value", new_dict[sb][4] * 1000)
                error = (np.sqrt(full[sb][3] ** (-2) +
                         new_dict[sb][3] ** (-2)) ** (-1))
                # TODO: unify average value calculations into function calls
                avg = (
                    full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] /
                    (new_dict[sb][3] ** 2)) / (
                    full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                full[sb][2] = avg
                full[sb][3] = error

                lw_error = (np.sqrt(full[sb][5] ** (-2) +
                            new_dict[sb][5] ** (-2)) ** (-1))
                lw_avg = (
                    full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] /
                    (new_dict[sb][5] ** 2)) / (
                    full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                # This may not be the exactly right way to calculate the error
                full[sb][5] = lw_error
                if verbose:
                    print("New value", lw_avg * 1000)
        except Exception:
            # I think this makes things work when there's no overlap
            new_starter = 0
    if verbose:
        print("appending new elements. new_starter={}".format(new_starter))

    # This loop will add the sidebands which were only seen in the second step
    for sb in [x for x in list(new_dict.keys()) if (
     (x >= new_starter) or (x in missing))]:
        full[sb] = new_dict[sb]
        if need_ratio:
            full[sb][2] = ratio * full[sb][2]
            full[sb][3] = full[sb][2] * np.sqrt(
                (error / ratio) ** 2 +
                (ratio * full[sb][3] / full[sb][2]) ** 2)
            # print '\n2164\nfull[2]', full[0][2]
    if verbose:
        print("I made this dictionary", sorted(full.keys()))
    return full


def stitch_hsg_dicts(
 full_obj, new_obj, need_ratio=False, verbose=False, ratios=[1, 1],
 override_ratio=False, ignore_weaker_lowers=True):
    """
    This helper function takes a FullHighSideband and a sideband object, either
    CCD or PMT and smushes the new sb_results into the full_dict.

    The first input doesn't change, so f there's a PMT set of data involved, it
    should be in the full variable to keep the laser normalization intact.

    This function almost certainly does not work for stitching many negative
    orders in it's current state

    11/14/16
    --------
    This function has been updated to take the CCD objects themselves to be
    more intelligent about stitching. Consider two scans, (a) spec step 0 with
    1 gain, spec step 2 with 110 gain and (b) spec step 0 with 50 gain and spec
    step 1 with 110 gain. The old version would always take spec step 0 to
    scale to, so while comparisons between spec step 0 and 1 for either case is
    valid, comparison between (a) and (b) were not, since they were scaled to
    different gain parameters. This new code will check what the gain values
    are and scale to the 110 data set, if present. This seems valid because we
    currently always have a 110 gain exposure for higher order sidebands. The
    exception is if the laser is present (sideband 0), as that is an absolute
    measure to which all else should be related.
    TODO: run some test cases to test this.

    06/11/18
    --------
    That sometimes was breaking if there were only 3-4 sidebands to fit with
    poor SNR. I've added the override_ratio to be passed to set a specific
    ratio to scale by. From data on 06/03/18, the 50gain to 110gain is a ~3.6
    ratio. I haven't done a clean way of specifying which data set it should be
    scaled. Right now, it leaves the laser line data, or the 110 gain data
    alone.


    Inputs:
    full = full_dict from FullHighSideband, or HighSidebandPMT.  It's important
           that it contains lower orders than the new_dict.
    new_dict = another full_dict.
    need_ratio = If gain or other parameters aren't equal and must resort to
                 calculating the ratio instead of the measurements being
                 equivalent. Changing integration time still means N photons
                 made M counts, but changing gain or using PMT or whatever does
                 affect things.
    ratios: Will update with the values to the ratios needed to scale the data.
            ratios[0] is the ratio for the "full_obj"
            ratios[1] is the ratio for the "new_obj"
            one of them will be one, one will be the appropriate scale, since
            one of them is unscaled. This is strictly speaking an output
    override_ratio: Pass a float to specify the ratio that should be used.
    ignore_weaker_lowers: Sometimes, a SB is in the short pass filter so a
        lower order is weaker than the next highest. If True, causes script to
        ignore all sidebands which are weaker and lower order.

    Returns:
    full = extended version of the input full.  Overlapping sidebands are
           averaged because that makes sense?
    """
    if isinstance(full_obj, dict) and isinstance(new_obj, dict):
        return stitch_hsg_dicts_old(full_obj, new_obj, need_ratio, verbose)

    if verbose:
        print("=" * 15)
        print()
        print("Stitching HSG dicts")
        print()
        print("=" * 15)

    # remove potentially offensive SBs, i.e. a 6th order SB being in the SPF
    # for more data, but being meaningless to pull intensity information from.
    # Note: this might not be the best if you get to higher order stitches
    # where it's possible that the sidebands might not be monotonic
    # (from noise?)
    if ignore_weaker_lowers:
        full_obj.full_dict, full_obj.sb_results = (
            FullHighSideband.parse_sb_array(full_obj.sb_results))
        new_obj.new_dict, new_obj.sb_results = (
            FullHighSideband.parse_sb_array(new_obj.sb_results))

    # was messing around with references and causing updates to arrays when
    # it shouldn't be
    full = copy.deepcopy(full_obj.full_dict)
    new_dict = copy.deepcopy(new_obj.full_dict)

    # Force a rescaling if you've passed a specified parameter
    # if isinstance(override_ratio, float):
    #     need_ratio = True

    # Do some testing to see which dict should be scaled to the other
    # I honestly forget why I prioritized the PMT first like this. But the
    # third check looks to make a gain 110 prioritize non-110, unless the
    # non-110 includes a laser line
    scaleTo = ""
    # TODO: altar below elif knot into a function call for Mccabe and pylama
    if need_ratio:
        if isinstance(new_obj, HSPMT.HighSidebandPMT):
            scaleTo = "new"
        elif isinstance(full_obj, HSPMT.HighSidebandPMT):
            scaleTo = "full"
        # this line specifically requires the function treatment to correct
        elif new_obj.parameters["gain"] == 110 and \
          full_obj.parameters["gain"] != 110 and 0 not in full:
            scaleTo = "new"
        else:
            scaleTo = "full"

    if verbose:
        print("\tI'm adding these sidebands", sorted(new_dict.keys()))
        print("\t  With these:", sorted(full.keys()))
    # The list that hold which orders are in both dictionaries
    overlap = []
    # How to deal with sidebands that are missing from full but in new.
    missing = []
    for new_sb in sorted(new_dict.keys()):
        full_sbs = sorted(full.keys())
        if new_sb in full_sbs:
            overlap.append(new_sb)
        elif new_sb not in full_sbs and new_sb < full_sbs[-1]:
            # This probably doesn't work with bunches of negative orders
            missing.append(new_sb)

    if verbose:
        print("\t  ( overlap:", overlap, ")")
        print("\t  ( missing:", missing, ")")

    # This if-else clause handles how to average together overlapping sidebands
    # which are seen in both spectra,
    if need_ratio:
        # Calculate the appropriate ratio to multiply the new sidebands by.
        # I'm not entirely sure what to do with the error of this guy.
        ratio_list = []
        try:
            new_starter = overlap[-1]
            if verbose:
                print("\n\tadding these ratios,", end=' ')
            # TODO: code below appears highly redundant with stitch_hsg_dicts
            #   and thus is prime for conversion to a function call
            if len(overlap) > 2:
                overlap = [x for x in overlap if (x % 2 == 0)]
                # and (x != min(overlap) and (x != max(overlap)))]
            if scaleTo == "new":
                if verbose:
                    print("scaling to new :")
                for sb in overlap:
                    ratio_list.append(new_dict[sb][2]/full[sb][2])
                    if verbose:
                        print("\t\t{:2.0f}: {:.3e}/{:.3e} ~ {:.3e},".format(
                            sb, new_dict[sb][2], full[sb][2], ratio_list[-1]))
                # new_ratio = 1 06/11/18 Not sure what these were used for
                ratio = np.mean(ratio_list)
            else:
                if verbose:
                    print("scaling to full:")
                for sb in overlap:
                    ratio_list.append(full[sb][2] / new_dict[sb][2])
                    if verbose:
                        print("\t\t{:2.0f}: {:.3e}/{:.3e} ~ {:.3e},".format(
                            sb, full[sb][2], new_dict[sb][2], ratio_list[-1]))

                # 06/11/18 Not sure what these were used for
                # new_ratio = np.mean(ratio_list)

                ratio = np.mean(ratio_list)
            # Maybe not the best way to do it, performance wise, since you
            # still iterate through the list, even though you'll override it.
            if isinstance(override_ratio, float):
                ratio = override_ratio
                if verbose:
                    print("overriding calculated ratio with user inputted")
            error = np.std(ratio_list) / np.sqrt(len(ratio_list))

        except IndexError:
            # If there's no overlap (which you shouldn't let happen), hardcode
            # a ratio and error. I looked at all the ratios for the overlaps
            # from 6/15/16 (540ghz para) to get the rough average. Hopefully
            # they hold for all data.
            if not overlap:
                ratio = 0.1695
                error = 0.02
                # no overlap, so make sure it grabs all the sidebands
                new_starter = min(new_dict.keys())
            else:
                raise
        if verbose:
            # print "Ratio list\n\t", ("{:.3g}, "*len(ratio_list))[:-2].format(
            # *ratio_list)
            # print "Overlap   \n\t", [round(ii, 3) for ii in overlap]
            print("\t Ratio: {:.3g} +- {:.3g} ({:.2f}%)\n".format(
                ratio, error, error/ratio*100))
        # Adding the new sidebands to the full set and moving errors around.
        # I don't know exactly what to do about the other aspects of the
        # sidebands besides the strength and its error.
        if scaleTo == "full":
            ratios[1] = ratio
            for sb in overlap:
                if verbose:
                    print("For SB {:02d}, original strength is {:.3g} +- {:.3g} ({:.3f}%)".format(int(sb), new_dict[sb][2], new_dict[sb][3], new_dict[sb][3]/new_dict[sb][2]*100 ))

                new_dict[sb][3] = \
                    ratio * new_dict[sb][2] * np.sqrt(
                    (error / ratio) ** 2 +
                    (new_dict[sb][3] / new_dict[sb][2]) ** 2)
                new_dict[sb][2] = ratio * new_dict[sb][2]
                if verbose:
                    print("\t\t   scaled\t\t\t\t{:.3g} +- {:.3g} ({:.3f}%)".format(new_dict[sb][2], new_dict[sb][3], new_dict[sb][3]/new_dict[sb][2]*100))
                    print("\t\t   full\t\t\t\t\t{:.3g} +- {:.3g} ({:.3f}%)".format(full[sb][2], full[sb][3], full[sb][3]/full[sb][2]*100))

                sb_error = np.sqrt(full[sb][3] ** (-2) +
                                   new_dict[sb][3] ** (-2)) ** (-1)

                avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (
                    new_dict[sb][3] ** 2)) / (full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                full[sb][2] = avg
                full[sb][3] = sb_error
                if verbose:
                    print("\t\t   replaced with \t\t{:.3g} +- {:.3g} ({:.3f}%)".format(full[sb][2], full[sb][3], full[sb][3]/full[sb][2]*100))
                    print()

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (
                new_dict[sb][5] ** 2)) / (
                             full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error
        else:
            ratios[0] = ratio
            for sb in overlap:
                full[sb][3] = ratio * full[sb][2] * np.sqrt((error / ratio) ** 2 + (full[sb][3] / full[sb][2]) ** 2)
                full[sb][2] = ratio * full[sb][2]

                sberror = np.sqrt(full[sb][3] ** (-2) + new_dict[sb][3] ** (-2)) ** (-1)
                avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (
                    new_dict[sb][3] ** 2)) / (full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                full[sb][2] = avg
                full[sb][3] = sberror

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (
                new_dict[sb][5] ** 2)) / (
                             full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error


    else: # not needing a new ratio
        try:
            new_starter = overlap[-1] # This grabs the sideband order where only the new dictionary has
                                      # sideband information.  It's not clear why it necessarily has to be
                                      # at this line.
            overlap = [x for x in overlap if (x % 2 == 0)
                       ] # and (x != min(overlap) and (x != max(overlap)))]
            # This cuts out the lowest order sideband in the overlap for mysterious reasons
            for sb in overlap: # This for loop average two data points weighted by their relative errors
                if verbose:
                    print("The sideband", sb)
                    print("Old value", full[sb][4] * 1000)
                    print("Add value", new_dict[sb][4] * 1000)
                try:
                    error = np.sqrt(full[sb][3] ** (-2) + new_dict[sb][3] ** (-2)) ** (-1)
                    avg = (full[sb][2] / (full[sb][3] ** 2) + new_dict[sb][2] / (new_dict[sb][3] ** 2)) / (
                        full[sb][3] ** (-2) + new_dict[sb][3] ** (-2))
                    full[sb][2] = avg
                    full[sb][3] = error
                except RuntimeWarning:
                    raise IOError()

                lw_error = np.sqrt(full[sb][5] ** (-2) + new_dict[sb][5] ** (-2)) ** (-1)
                lw_avg = (full[sb][4] / (full[sb][5] ** 2) + new_dict[sb][4] / (new_dict[sb][5] ** 2)) / (
                full[sb][5] ** (-2) + new_dict[sb][5] ** (-2))
                full[sb][4] = lw_avg
                full[sb][5] = lw_error  # This may not be the exactly right way to calculate the error
                if verbose:
                    print("New value", lw_avg * 1000)
        except:
            new_starter = 0  # I think this makes things work when there's no overlap
    if verbose:
        print("appending new elements. new_starter={}".format(new_starter))


    for sb in [x for x in list(new_dict.keys()) if ((x > new_starter) or (x in missing))]:
        full[sb] = new_dict[sb]
        if scaleTo == "full":
            full[sb][2] = ratio * full[sb][2]
            full[sb][3] = full[sb][2] * np.sqrt((error / ratio) ** 2 + (ratio * full[sb][3] / full[sb][2]) ** 2)
    if scaleTo == "new":
        for sb in set(full.keys()) - set(sorted(new_dict.keys())[:]):
            full[sb][2] *= ratio
            # TODO: I think this is an invalid error
            # propagation (since ratio has error associated with it
            full[sb][3] *= ratio
    if verbose:
        print("I made this dictionary", sorted(full.keys()))
        print('-'*19)
        return full
        return full, ratio #the fuck? Why was this here?

    return full
