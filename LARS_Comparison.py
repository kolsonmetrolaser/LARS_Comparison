import numpy as np
import scipy.signal as sig
import os
import pathlib
import LarsDataClass
from LarsDataClass import LarsData
import plotfunctions as pf
from time import time
from numpy.typing import ArrayLike, NDArray
from filters import airpls, sgf
from helpers import group
from typing import Literal, Iterable
from itertools import combinations
import pickle
from needlemanwunsch import find_matches

np.set_printoptions(precision=3)


def remove_baseline(y: ArrayLike, **settings) -> tuple[ArrayLike, ArrayLike]:
    """
    Removes the baseline, as calculculated by `airpls`, from spectral data.

    Parameters
    ----------
    y : ArrayLike
        The data to remove the baseline from.
    settings : optional kwargs
        See `main()`.

    Returns
    -------
    y_baseline_removed : ArrayLike
        The data with the baseline removed.
    baseline : ArrayLike
        The baseline.

    """
    baseline_smoothness = settings['baseline_smoothness'] if 'baseline_smoothness' in settings else 1e12
    baseline_polyorder = settings['baseline_polyorder'] if 'baseline_polyorder' in settings else 2
    baseline_itermax = settings['baseline_itermax'] if 'baseline_itermax' in settings else 10

    baseline = airpls(y, lam=baseline_smoothness, porder=baseline_polyorder, itermax=baseline_itermax)
    return y-baseline, baseline


def remove_noise(y: ArrayLike, normalize: bool = True, noise: Literal[None, ArrayLike] = None)\
        -> tuple[ArrayLike, float]:
    """
    Removes the noise, either supplied as `noise` or the root mean square of `y`, from `y`.

    Parameters
    ----------
    y : ArrayLike
        The data to remove noise from.
    normalize : bool, optional
        Whether to normalize the data. The default is True.
    noise : Literal[None,ArrayLike], optional
        The noise level. Can be given as a single value or an array the same size as `y`.
        The default is uses the root mean square of `y`.

    Returns
    -------
    ArrayLike
        The data with noise removed.
    float
        The noise level.
    """
    y = y.copy()
    noise = np.sqrt(np.mean(y**2)) if noise is None else noise
    if normalize:
        y /= noise
        y -= 1
    else:
        y -= noise
    y[y < 0] = 0
    return y, noise


def fit_peaks(x: ArrayLike, y: ArrayLike, **settings) -> dict:
    """
    Finds the location of peaks in `y` along the `x` axis.
    In spectra, `x` and `y` typically refer to the frequency and amplitude, respectively.

    Parameters
    ----------
    x : ArrayLike
        Evenly spaced x-axis data. Typically the frequency.
    y : ArrayLike
        y-axis data. Typically the amplitude.
    settings : optional kwargs
        See `main()`.

    Returns
    -------
    dict
        A dictionary of peak data, containing the keys:
            'count': Number of peaks
            'indices': Indices of peaks
            'positions': The locations of peaks in `x`
            'heights': The heights of peaks in `y` units
            'widths': The widths of peaks in `x` units. Assumes `x` is evenly spaced.
            'lefts', 'rights': The left and right edges of peaks in `x` units.
                               Assumes `x` is evenly spaced.

    """

    peak_height_min = settings['peak_height_min'] if 'peak_height_min' in settings else 0.2
    peak_prominence_min = settings['peak_prominence_min'] if 'peak_prominence_min' in settings else 0.2
    peak_ph_ratio_min = settings['peak_ph_ratio_min'] if 'peak_ph_ratio_min' in settings else 0.5

    d = {}
    peaks = sig.find_peaks(y, height=peak_height_min, prominence=peak_prominence_min)
    peak_widths = sig.peak_widths(y, peaks[0], rel_height=1,
                                  prominence_data=(peaks[1]['prominences'],
                                                   peaks[1]['left_bases'],
                                                   peaks[1]['right_bases']))

    d['indices'] = np.array([i for i, (h, p) in
                             enumerate(zip(peaks[1]['peak_heights'], peaks[1]['prominences']))
                             if p/h > peak_ph_ratio_min])
    d['count'] = len(d['indices'])
    d['positions'], d['heights'], d['widths'] =\
        x[peaks[0][d['indices']]], peaks[1]['peak_heights'][d['indices']], peak_widths[0][d['indices']]*(x[1]-x[0])

    d['rights'] = d['positions'] + d['widths']/2
    d['lefts'] = d['positions']-d['widths']/2
    return d


def detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed,
                   vels_filtered, xlim=[10, 60], ylim=[0, 10], vels_peaks_removed=None):
    """
    Makes optional detailed plots inside `analyze_data()`.
    """
    kwargs = {'x_label': 'Frequency (kHz)', 'vlinewidth': 1}
    pf.line_plot(freqs/1000, [vels], style='.', x_lim=xlim,
                 title=f'{folder}{name} raw data', y_lim=[-2, 200], **kwargs, y_label='Amplitude (μm/s)')
    pf.line_plot(freqs/1000, [vels, vels-vels_baseline_removed], style='.', x_lim=xlim,
                 title=f'{folder}{name} raw data and baseline', y_lim=[-2, 200], **kwargs, y_label='Amplitude (μm/s)')
    if vels_peaks_removed is not None:
        pf.line_plot(freqs/1000, [vels_peaks_removed, vels-vels_baseline_removed], style='.', x_lim=xlim,
                     title=f'{folder}{name} peaks removed and baseline', y_lim=[-2, 70],
                     **kwargs, y_label='Amplitude (μm/s)')
    pf.line_plot(freqs/1000, [vels_baseline_removed], style='.', x_lim=xlim,
                 title=f'{folder}{name} baseline removed', y_lim=[-2, 200], **kwargs, y_label='Amplitude (μm/s)')
    pf.line_plot(freqs/1000, [vels_rms_norm_zeroed], style='.', x_lim=xlim,
                 title=f'{folder}{name} rms_norm_zero', y_lim=[0, 15], **kwargs, y_label='Amplitude (arb.)')
    pf.line_plot(freqs/1000, [vels_filtered], style='.', x_lim=xlim,
                 title=f'{folder}{name} filtered data', y_lim=ylim, **kwargs, y_label='Amplitude (arb.)')
    pf.line_plot(freqs/1000, [vels_filtered], style='.', x_lim=xlim, v_line_pos=peaks['positions']/1000,
                 title=f'{folder}{name} peak fits', y_lim=ylim, **kwargs, y_label='Amplitude (arb.)')
    return


def analyze_data(data: LarsData, **settings) -> tuple[dict, NDArray, NDArray, NDArray, str]:
    """
    Smooths raw LARS data with `filters.sgf()`, performs background removal with `filters.airpls()`,
    and fits peaks with `scipy.signal.find_peaks()`.

    For `sgf` information, see https://medium.com/pythoneers/introduction-to-the-savitzky-golay-filter-a-comprehensive-guide-using-python-b2dd07a8e2ce.
    For `airpls` information, see https://code.google.com/archive/p/airpls/

    Parameters
    ----------
    data : LarsData
        LarsData to be analyzed.
    settings : optional kwargs
        See `main()`.

    Returns
    -------
    peaks : dict
        A dictionary of peak data, containing the keys:
            'count': Number of peaks
            'indices': Indices of peaks
            'positions': The locations of peaks in `x`
            'heights': The heights of peaks in `y` units
            'widths': The widths of peaks in `x` units. Assumes `x` is evenly spaced.
            'lefts', 'rights': The left and right edges of peaks in `x` units. Assumes `x` is evenly spaced.
    freqs : NDArray
        Frequencies corresponding to the velocities.
    vels : NDArray
        Velocities before smoothing and baseline correction.
    newvels : NDArray
        Velocities after smoothing and baseline correction.
    name: str
        Name of the data.

    """
    slc_limits = settings['slc_limits'] if 'slc_limits' in settings else (12000, 60000)
    plot = settings['plot'] if 'plot' in settings else False
    plot_detail = settings['plot_detail'] if 'plot_detail' in settings else True
    plot_recursive_noise = settings['plot_recursive_noise'] if 'plot_recursive_noise' in settings else True
    recursive_noise_reduction = settings['recursive_noise_reduction'] if\
        'recursive_noise_reduction' in settings else True
    max_noise_reduction_iter = settings['max_noise_reduction_iter'] if 'max_noise_reduction_iter' in settings else 10
    frange = settings['frange'] if 'frange' in settings else (0, 200)
    sgf_applications = settings['sgf_applications'] if 'sgf_applications' in settings else 2
    sgf_windowsize = settings['sgf_windowsize'] if 'sgf_windowsize' in settings else 101
    sgf_polyorder = settings['sgf_polyorder'] if 'sgf_polyorder' in settings else 0
    peak_plot_width = settings['peak_plot_width'] if 'peak_plot_width' in settings else 10
    regularization_ratio = settings['regularization_ratio'] if 'regularization_ratio' in settings else 0.5

    freqs = data.freq
    slc = np.logical_and(freqs > slc_limits[0], freqs < slc_limits[1])
    freqs = freqs[slc]
    vels = data.vel[slc]
    name = data.name
    folder = pathlib.Path(data.path).parts[-2]

    vels_baseline_removed, baseline = remove_baseline(vels, **settings)

    vels_rms_norm_zeroed, noise = remove_noise(vels_baseline_removed)

    vels_filtered = sgf(vels_rms_norm_zeroed, n=sgf_applications, w=sgf_windowsize, p=sgf_polyorder)

    peaks = fit_peaks(freqs, vels_filtered, **settings)

    if plot and ((plot_detail and not recursive_noise_reduction)
                 or (recursive_noise_reduction and plot_recursive_noise)):
        detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed, vels_filtered)

    recursive_noise_iterations = 0
    while recursive_noise_reduction:
        noise_prev = noise
        vels_peaks_removed = vels.copy()
        vels_peaks_removed_for_baseline = vels.copy()
        for left, right in zip(peaks['lefts'], peaks['rights']):
            vels_peaks_removed[np.logical_and(freqs > left, freqs < right)] = np.nan
            vels_peaks_removed_for_baseline[np.logical_and(
                freqs > left, freqs < right)] = baseline[np.logical_and(freqs > left, freqs < right)]
        # freqs_peaks_removed = freqs[np.logical_not(np.isnan(vels_peaks_removed))]
        indices_nonpeak = np.logical_not(np.isnan(vels_peaks_removed))
        if not np.any(indices_nonpeak):
            print("""Removed all the data during iterative noise calculations.
            This is not ideal, but not necessarily a problem.""")
            break
        vels_peaks_removed = vels_peaks_removed[indices_nonpeak]

        _, baseline_peaks_removed = remove_baseline(vels_peaks_removed_for_baseline, **settings)
        baseline = baseline_peaks_removed.copy()

        updated_baseline = baseline.copy()

        vels_peaks_removed_baseline_removed = vels_peaks_removed - baseline[indices_nonpeak]

        rms = np.sqrt(np.mean(vels_peaks_removed_baseline_removed**2))

        vels_baseline_removed = vels.copy() - updated_baseline

        noise = regularization_ratio*rms + (1-regularization_ratio)*noise_prev

        vels_rms_norm_zeroed, noise = remove_noise(vels_baseline_removed, noise=noise)

        vels_filtered = sgf(vels_rms_norm_zeroed, n=sgf_applications, w=sgf_windowsize, p=sgf_polyorder)

        peaks_updated = fit_peaks(freqs, vels_filtered, **settings)

        if 'PRINT_MODE' in settings and settings['PRINT_MODE'] == 'full':
            print(f'{peaks['count']} peaks to {peaks_updated['count']} peaks')

        recursive_noise_iterations += 1
        if recursive_noise_iterations >= max_noise_reduction_iter:
            if 'PRINT_MODE' in settings and settings['PRINT_MODE'] in ['sparse', 'full']:
                print('Reached maximum recursive noise iterations for', name)
        recursive_noise_reduction = (not peaks['count'] == peaks_updated['count']) and (
            not recursive_noise_iterations >= max_noise_reduction_iter)
        peaks = peaks_updated

        if plot and plot_detail and plot_recursive_noise:
            detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed,
                           vels_filtered, vels_peaks_removed=vels_peaks_removed_for_baseline)

    if plot and plot_detail and recursive_noise_reduction and not plot_recursive_noise:
        detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed, vels_filtered)

    newvels = vels_filtered

    if 'PRINT_MODE' in settings and settings['PRINT_MODE'] == 'full':
        peaklist = peaks['positions'][np.logical_and(peaks['positions'] > frange[0]*1000,
                                                     peaks['positions'] < frange[1]*1000)]
        print(f"""for {folder}/{name}
              peaks at:     {peaklist/1000} kHz""")

    if plot:
        peak_groups = group(peaks['positions'][np.logical_and(peaks['positions'] > frange[0]
                            * 1000, peaks['positions'] < frange[1]*1000)]/1000, peak_plot_width)

        pf.line_plot(freqs/1000, [vels, newvels], style='.', x_lim=frange, v_line_pos=peaks['positions']/1000,
                     vlinewidth=1, title=f'{folder}{name} peak fit', y_norm='each')
        for pg in peak_groups:
            if np.size(pg) > 1:
                avg_pos = (pg[0]+pg[-1])/2
                xl = [avg_pos-peak_plot_width/2, avg_pos+peak_plot_width/2]
            else:
                xl = [pg-peak_plot_width/2, pg+peak_plot_width/2]
            pf.line_plot(freqs/1000, [vels, newvels], style='.', x_lim=xl, v_line_pos=peaks['positions']/1000,
                         vlinewidth=1, title=f'{folder}{name} peak fit', y_norm='each')

    return peaks, freqs, vels, newvels, name


def LARS_analysis(folder: str = '', previously_loaded_data: Literal[None, LarsData] = None, **settings)\
        -> tuple[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike], LarsData]:
    """
    Caches LARS data locally, if appropriate and handles walking through file structures within `folder`.

    Combines data in `folder`

    Largely a wrapper for `analyze_data`, see that function for more detail about `frange` and `plot`.

    Parameters
    ----------
    folder : str, optional
        Path to the folder containing `.all` LARS data files, or a folder tree containing such files. The default is ''.
    previously_loaded_data : Literal[None,LarsData], optional
        LarsData object of loaded data that will be analyzed. The default is None.
    settings : optional kwargs
        See `main()`.

    Returns
    -------
    tuple[tuple[ArrayLike,ArrayLike,ArrayLike,ArrayLike,ArrayLike],LarsData]
        tuple[ArrayLike,ArrayLike,ArrayLike,ArrayLike,ArrayLike]
            See `analyze_data()` for the first return values.
            The final return value is the LarsData object corresponding to `folder` with mode `combine`.
        LarsData
            LarsData object corresponding to `folder` with mode `combine`.

    """
    if folder == '':
        return None

    if previously_loaded_data is None:
        # Load data
        alldata = []
        for subdir, dirs, files in os.walk(folder):
            for file in [f for f in files if '.all' in f]:
                alldata.append(LarsData.from_file(os.path.join(subdir, file)))

        # Analyze data
        combine = settings['combine'] if 'combine' in settings else 'max'
        if combine is not None:
            data_to_analyze = LarsDataClass.combine(alldata, combine)
            # analysis = analyze_data(combined_data,frange=frange,plot=plot)
        else:
            return None
    else:
        print(f'Using previously loaded data for {previously_loaded_data.name}')
        data_to_analyze = previously_loaded_data
    analysis = analyze_data(data_to_analyze, **settings)
    data_to_analyze.newvel = analysis[3]
    data_to_analyze.peaks = analysis[0]
    data_to_analyze.analyzed_this_session = True

    return analysis, data_to_analyze


def compare_LARS_measurements(folders: Iterable = [], previously_analyzed_data: tuple = (None, None), **settings)\
        -> tuple[dict, list[LarsData]]:
    """
    Compare two LARS measurements, each stored in folders containing LARS data from
    multiple points stored in whitespace-deliminated `.all` files.
    The first folder in `folders` is the reference, and the second is the measurement. The second is allowed to stretch.

    Parameters
    ----------
    folders : Iterable, optional
        Group of two folders to compare. Each `folder` in the `folders` list should give a full file path.
        The default is [].
    previously_analyzed_data : tuple, optional
        Previously analyzed data corresponding to `folders`.
        Either or both may be `None`, which will load and combined by this function.
    settings : optional kwargs
        See `main()`.
    Returns
    -------
    tuple[dict,list[LarsData]] :
        dict
            matching_analysis = {'graph':best_graph,'stretch':best_stretch,'quality':best_quality,
                                 'names':names,'matched':matched,'unmatched':[unmatched_X,unmatched_Y]}
            A dictionary of matching analysis data, containing the keys:
                'graph': The BipartiteGraph object containing the matches between peaks.
                'stretch': The stretching factor
                'quality': The quality of the match. Lower is better.
                'names': The names of the two data sets in the order [reference,measurement]
                'matched': Frequencies that the matched peaks were found at
                'unmatched': The frequencies of unmatched peaks in each data set in the order [reference,measurement]
                'match_probability': Probability a peak is matched. Equal to `2*m/(2*m+u_r+u_m)` where
                    m is the number of matched peaks,and u_r and u_m are the number of unmatched peaks
                    in the reference and measured data, respectively.
        list[LarsData]
            List of LarsData objects corresponding to `folder` with mode `combine`.

    """
    if len(folders) != 2:
        return False

    max_stretch = settings['max_stretch'] if 'max_stretch' in settings else 0.02
    num_stretches = settings['num_stretches'] if 'num_stretches' in settings else 1000
    stretching_iterations = settings['stretching_iterations'] if 'stretching_iterations' in settings else 5
    stretch_iteration_factor = settings['stretch_iteration_factor'] if 'stretch_iteration_factor' in settings else 5
    matching_penalty_order = settings['matching_penalty_order'] if 'matching_penalty_order' in settings else 1
    peak_match_window = settings['peak_match_window'] if 'peak_match_window' in settings else 150
    nw_normalized = settings['nw_normalized'] if 'nw_normalized' in settings else False

    # collect peak positions, and the frequency, raw velocity, and smoothed velocity vectors from each folder
    positions = []
    names = []
    freqs = []
    vels = []
    newvels = []
    datas = []
    for i, f in enumerate(folders):
        if previously_analyzed_data[i] is None:
            print(f'Loading and analyzing data from {f}')
            # (peaks, freq, vel, newvel, name), data = LARS_analysis(folder=f,combine=combine,frange=(10,60),plot=plot)
            (peaks, freq, vel, newvel, name), data = LARS_analysis(folder=f, **settings)
        else:
            if hasattr(previously_analyzed_data[i], 'analyzed_this_session') and\
                    previously_analyzed_data[i].analyzed_this_session:
                print(f'Using previously analyzed data for {previously_analyzed_data[i].name}')
                pad = previously_analyzed_data[i]
                (peaks, freq, vel, newvel, name), data = (pad.peaks, pad.freq, pad.vel, pad.newvel, pad.name), pad
            else:
                (peaks, freq, vel, newvel, name), data =\
                    LARS_analysis(folder=f, previously_loaded_data=previously_analyzed_data[i], **settings)
        positions.append(peaks['positions'])
        names.append(name)
        freqs.append(freq)
        vels.append(vel)
        newvels.append(newvel)
        datas.append(data)

    kwargs_find_matches = {'max_stretch': max_stretch, 'num_stretches': num_stretches,
                           'stretching_iterations': stretching_iterations,
                           'stretch_iteration_factor': stretch_iteration_factor,
                           'penalty_order': matching_penalty_order,
                           'gap': peak_match_window/2, 'nw_normalized': nw_normalized}
    bestrx, bestry, best_quality, best_stretch, search_space_delta = find_matches(
        positions[0], positions[1], **kwargs_find_matches)

    # print results
    if 'PRINT_MODE' in settings and settings['PRINT_MODE'] == 'full':
        print(f'Best matches found with quality {best_quality:.5f} at stretch {
              best_stretch:.5f} ± {search_space_delta:.5f}')
        for x, y in zip(bestrx, bestry):
            print(f'{x:7.1f}  {best_stretch*y:7.1f}')
    unmatched_X = [x/1000 for x, y in zip(bestrx, bestry) if y == -1]
    unmatched_Y = [y/1000 for x, y in zip(bestrx, bestry) if x == -1]
    matched = [(x+y)/2/1000 for x, y in zip(bestrx, bestry) if x != -1 and y != -1]
    if 'PRINT_MODE' in settings and settings['PRINT_MODE'] == 'full':
        print(f'{len(unmatched_X)} unmatched peaks in reference at {unmatched_X} kHz')
        print(f'{len(unmatched_Y)} unmatched peaks in measurement at {unmatched_Y} kHz')
        print(f'{len(matched)} matched peaks at {matched} kHz')

    # Plot results
    # if plot:
    if 'plot' in settings and settings['plot']:
        xlims = ([10, 10+50/3], [10+50/3, 10+100/3], [10+100/3, 10+150/3])
        # for xlim in xlims:
        #     pf.line_plot([freqs[0]/1000, freqs[1]/1000], [newvels[0], newvels[1]], style='.',
        #                  x_lim=xlim, v_line_pos=[positions[0]/1000, positions[1]/1000],
        #                  v_line_color=['C0', 'C1'], vlinewidth=[2, 2], y_norm='each', title='Unstretched peak fits')
        # for xlim in xlims:
        #     pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [newvels[0], newvels[1]], style='.',
        #                  x_lim=xlim, v_line_pos=[positions[0]/1000, positions[1]/1000*best_stretch],
        #                  v_line_color=['C0', 'C1'], vlinewidth=[2, 2], y_norm='each', title='Stretched peak fits')
        # for xlim in xlims:
        #     pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [newvels[0], newvels[1]], style='.',
        #                  x_lim=xlim, v_line_pos=[matched, unmatched_X, unmatched_Y], v_line_color=['k', 'C0', 'C1'],
        #                  vlinewidth=[4, 2, 2], y_norm='each', title='Stretched peak matches filtered')
        for xlim in xlims+([30, 34],):
            pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [vels[0], vels[1]], style='.', x_lim=xlim,
                         v_line_pos=[matched, unmatched_X, unmatched_Y], v_line_color=['k', 'C0', 'C1'],
                         vlinewidth=[4, 2, 2], y_norm='each', title='Stretched peak matches raw')

    matching_analysis = {'stretch': best_stretch, 'quality': best_quality, 'name': names, 'matched': matched,
                         'unmatched': [unmatched_X, unmatched_Y],
                         'match_probability': 2*len(matched)/(2*len(matched)+len(unmatched_X)+len(unmatched_Y))}

    return matching_analysis, datas


def analyze_each_pair_of_folders(folders: Iterable = [], **settings) -> tuple[list[dict], dict]:
    """
    Analyzes each possible pair from a group of folders.

    Loading data is slow, so that should be updated, though...

    Parameters
    ----------
    folders : Iterable, optional
        Group of folders to analyze. The default is [].
    settings : optional kwargs
        See `main()`.

    Returns
    -------
    list[dict]
        List of matching results for each pair. See `compare_LARS_measurements()`.
    dict
        dict containing raw and processed data for each folder. See `compare_LARS_measurements()`.

    """
    if len(folders) < 2:
        return False

    results = []
    if 'pickled_data_path' in settings and settings['pickled_data_path'] != '':
        print(f'Loading data from {settings['pickled_data_path']}')
        with open(settings['pickled_data_path'], 'rb') as inp:
            data_dict = pickle.load(inp)
    else:
        data_dict = {}

    folder_pairs = list(combinations(folders, 2))
    for i, fpair in enumerate(folder_pairs):
        if 'PRINT_MODE' in settings and settings['PRINT_MODE'] in ['sparse', 'full']:
            print(f"""Analyzing    {os.path.split(fpair[0])[1]}    and    {
                  os.path.split(fpair[1])[1]}    (pair {i+1} of {len(folder_pairs)})""")
        data_0 = data_dict[fpair[0]] if fpair[0] in data_dict.keys() else None
        data_1 = data_dict[fpair[1]] if fpair[1] in data_dict.keys() else None
        matching_analysis, datas = compare_LARS_measurements(
            fpair, previously_analyzed_data=(data_0, data_1), **settings)
        results.append(matching_analysis)
        for data in datas:
            if data.path not in data_dict.keys():
                data_dict[data.path] = data
    for datapath in data_dict:
        data_dict[datapath].analyzed_this_session = False
    return results, data_dict


def main():
    """
    Settings (Create dict called `settings` with the following keys.)
    ----------

        Data Definitions
        ----------------
            frange : tuple[float,float], optional
                Range of frequencies, in kHz, to plot over if `plot` is True. The default is (0,200).
            slc_limits : tuple, optional
                Frequency slice inside which peak fitting is performed.
            combine : Literal[None,'max','mean'], optional
                How to combine the data in `folder`. If None, return None.
                If `max` or `mean`, combine data. See `LarsDataClass.combine` for detail. The default is None.

        Plotting and Printing
        ---------------------
            plot : bool, optional
                Whether to create plots. The default is True.
            plot_detail : bool, optional
                Whether to create plots of fitting details. The default is False.
            plot_recursive_noise : bool, optional
                Whether to plot each iteration of recursive noise removal. Only applicable when
                `recursive_noise_reduction` is True. The default is False.
            peak_plot_width : float, optional
                Maximum width of individual plots if `plot` is True. If `peak_plot_width` is larger than
                the span of `frange`, multiple plots will be created. The default is 10.
            PRINT_MODE : str, optional
                'full' or 'sparse', depending on the output detail desired during analysis. The default is 'sparse'.

        Peak Fitting
        ------------
            Baseline Removal
            ----------------
                baseline_smoothness : float, optional
                    Larger `baseline_smoothness` results in a smoother baseline. The default is 1e12.
                baseline_polyorder : int, optional
                    The order of baseline differences of penalties. The default is 2.
                baseline_itermax : int, optional
                    Maximum number of iterations to find a baseline. Larger values may be more robust in certain cases,
                    but require more computational time. The default is 10.
            Smoothing
            ---------
                sgf_applications : int, optional
                    Number of recursive `scipy.signal.savgol_filter()` applications. The default is 2.
                sgf_windowsize: int, optional
                    The length of the smoothing filter window (i.e., the number of coefficients).
                    `sgf_windowsize` must be less than or equal to the size of the data. If it is even,
                    `1` is added automatically. The default is 101, equivalent to ±50 Hz with data spaced by 0.5 Hz.
                sgf_polyorder : int, optional
                    The order of the polynomial used to fit the samples during smoothing. `sgf_polyorder` must be
                    less than `sgf_windowsize`. The default is 0, roughly a smooth modified moving average.
            Peak Finding
            ------------
                peak_height_min : float, optional
                    Value in terms of a fraction of the noise level of the data which defines the minimum peak height
                    for consideration. The default is 0.01, correpsonding to a minimum height of `0.01*noise`
                    by setting `height` in `scipy.signal.find_peaks()`.
                peak_prominence_min : float, optional
                    Value in terms of a fraction of the noise level of the data which defines the minimum peak
                    prominence for consideration. The default is 0.01, correpsonding to a minimum prominence of
                    `0.01*noise` by setting `prominence` in `scipy.signal.find_peaks()`.
                peak_ph_ratio_min : float, optional
                    Minimum prominence-to-height ratio to accept a peak as valid. The default is 0.9.
            Noise Reduction
            ---------------
                recursive_noise_reduction : bool, optional
                    Whether to recursively remove peaks from data when the noise level is calculated.
                    The default is True.
                max_noise_reduction_iter: int, optional
                    Maximum number of recursive noise reduction iterations. The default is 10.
                regularization_ratio: float, optional
                    Fraction of the updated noise used to make the new noise in each iteration. The default is 0.5.

        Peak Matching
        -------------
            Stretching
            ----------
                max_stretch : float, optional
                    The measured data is allowed to stretch from `1-max_stretch` to `1+max_stretch`.
                    The default is 0.02.
                num_stretches : int, optional
                    Number of stretches to test per iteration. The default is 1000.
                stretching_iterations : int, optional
                    Number of stretch testing iterations. The default is 5.
                stretch_iteration_factor : float, optional
                    Factor by which the stretching search space is reduced each iteration. The default is 5.
            Matching
            --------
                peak_match_window : float, optional
                    The maximum allowed difference between matched peaks in Hz. The default is 150.
                matching_penalty_order : float, optional
                    Order of matching distance penalty. If 1, the penalty is the difference between two numbers.
                    If 2, the penalty is the square difference, etc. The default is 1.
        Saving
        ------
            save : bool, optional
                Whether to save the results in .pkl files. The default is True.
            save_tag : string, optional
                Additional tag to give the default names 'pair_results.pkl' and 'data_dict.pkl'. The default is ''.
            pickled_data_path : str, optional
                Path to pickled data_dict file. Useful when reanalyzing the same data with different settings.
                The default is None.
    """

    # MAKING DEFAULT SETTINGS
    # =======================
    settings = {}

    # DATA DEFINITIONS

    # Values, in kHz, of the data minimum and maximum frequency. Used for plotting and reporting.
    settings['frange'] = (10, 60)
    # Values, in Hz, of the data minimum and maximum frequency. Used for peak fitting and matching.
    settings['slc_limits'] = (10000, 60000)
    settings['combine'] = 'max'  # How to combine the data ('max' or 'mean')

    # PLOTTING AND PRINTING

    settings['plot'] = False  # Whether to create plots.
    settings['plot_detail'] = True  # Whether to create plots of fitting details
    settings['plot_recursive_noise'] = True  # Whether to plot each iteration of recursive noise removal.
    settings['peak_plot_width'] = 20  # Maximum width of some individual plots if `plot` is True.
    settings['PRINT_MODE'] = 'sparse'  # 'full' or 'sparse', depending on the output detail desired during analysis.

    # PEAK FITTING

    #    Baseline Removal

    settings['baseline_smoothness'] = 1e12  # Larger values give smoother baselines.
    settings['baseline_polyorder'] = 2  # The order of baseline differences of penalties.
    settings['baseline_itermax'] = 10  # Maximum airpls iterations to find a baseline.

    #    Smoothing

    settings['sgf_applications'] = 2  # Number of recursive `scipy.signal.savgol_filter()` applications
    # The length of the smoothing filter window (in data points, i.e., 101 = ±50 Hz window)
    settings['sgf_windowsize'] = 101
    settings['sgf_polyorder'] = 0  # The order of the polynomial used to fit the samples during smoothing.

    #    Peak Finding

    # Fraction of the noise level of the data which defines the minimum peak height for consideration.
    settings['peak_height_min'] = 0.2
    # Fraction of the noise level of the data which defines the minimum peak prominence for consideration.
    settings['peak_prominence_min'] = 0.2
    settings['peak_ph_ratio_min'] = 0.5  # Minimum prominence-to-height ratio to accept a peak as valid.

    #    Noise Reduction

    # Whether to recursively remove peaks from data when the noise level is calculated.
    settings['recursive_noise_reduction'] = True
    settings['max_noise_reduction_iter'] = 10  # Maximum number of recursive noise reduction iterations.
    # Fraction of the updated noise used to make the new noise in each iteration.
    settings['regularization_ratio'] = 0.5

    # PEAK MATCHING

    #    Stretching

    settings['max_stretch'] = 0.02  # The measured data is allowed to stretch from `1-max_stretch` to `1+max_stretch`.
    settings['num_stretches'] = 1000  # Number of stretches to test per iteration.
    settings['stretching_iterations'] = 10  # Number of stretch testing iterations.
    settings['stretch_iteration_factor'] = 5  # Factor by which the stretching search space is reduced each iteration.

    #    Matching

    # Maximum allowed difference between matched peaks.
    # Units are Hz if `nw_normalized` is False, else fraction of peak position.
    settings['peak_match_window'] = 150
    settings['matching_penalty_order'] = 1  # Order of matching distance penalty.
    settings['nw_normalized'] = False  # Whether to normalize the distance for the matching penalty.

    # SAVING

    settings['save'] = False  # Whether to save the results in .pkl files.
    # Additional tag to give the default names 'pair_results.pkl' and 'data_dict.pkl'.
    # They are saved as '[default_name]_[save_tag].pkl'.
    # settings['save_tag'] = '20241009_sgfapp4'
    settings['save_tag'] = ''  # Additional tag to give the default names 'pair_results.pkl' and 'data_dict.pkl'.
    # Path to pickled data_dict file. Useful when reanalyzing the same data with different settings.
    settings['pickled_data_path'] = ''

    # USERS
    # =======================
    # MODIFY SETTINGS
    # =======================
    # settings['PRINT_MODE'] = 'full'
    settings['pickled_data_path'] =\
        r'C:\Users\KOlson\OneDrive - Metrolaser, Inc\Documents\Python\LARS_Comparison\data_dict_10kHz.pkl'
    # settings['frange'] = (15,60)
    # settings['slc_limits'] = (15000,60000)
    # settings['save'] = False #Does not save results to .pkl files.
    # settings['peak_match_window'] = 0.0005
    # settings['nw_normalized'] = True # Whether to normalize the distance for the matching penalty.
    # settings['save_tag'] = '_nwnormal_0005'
    # settings['plot'] = True # Whether to create plots.

    # =======================
    # ENTER FOLDERS
    # =======================
    # Enter two folders with data to compare.
    # The folders should contain at least one `.all` file, which contains whitespace-deliminated data with 5 columns.
    # The 5 columns are:
    # 1. time (currently unused)
    # 2. piezoelectric transducor voltage (currently unused)
    # 3. laser doppler vibrometer voltage (currently unused)
    # 4. frequency (Hz)
    # 5. velocity (arbitrary unit, but must be the same between all files)

    # #Adityan Test
    # folders = [r'K:\MD05_2022_JDT\Data\20240930 Adityan test\x6 646',\
    #             r'K:\MD05_2022_JDT\Data\20240930 Adityan test\x6 655']

    # L Brackets
    # folders = [r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L1',\
    #             r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L2',\
    #             r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L3']

    # Many Brackets
    # according to file:///K:/DI02_2018_PhII_JT/Reporting/_Completed/DARPA%20PORTAL/Milestone%20%237_CLIN%200002%20Qtrly%2001/140D6318C0085_CLIN0002_Qtrly01.pdf
    # L1-L4 are defect-free
    # M: large defect, 1-3: bottom, 4-6: middle
    # N: medium defect, 1-3: bottom, 4-6: middle
    # O: small defect, 1-3: bottom, 4-6: middle
    # according to file:///K:/DI02_2018_PhII_JT/Reporting/_Completed/DARPA%20PORTAL/Milestone%20%2310_CLIN%200002%20Qtrly%2004/140D6318C0085%20CLIN%200002%20Qtrly04.pdf
    # Y5 is identical to L1 but printed with the P-X series
    # according to file:///K:/DI02_2018_PhII_JT/Reporting/_Completed/DARPA%20PORTAL/Milestone%20%2313_CLIN%200003%20Qtrly%2001/140D6318C0085%20TDI02JT13_CLIN%200003_Qtrly%2001.pdf
    # Y7 is identical to L1 but built 6 months later
    folders = [r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L1',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L2',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L3',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\L2',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M1',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M2',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M3',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M4',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M5',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N1',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N2',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N3',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N4',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N5',
               r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N6',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\O1',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\O2',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\O4',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\Y5',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\Y7']

    # CODE
    # MODIFY WITH CARE
    # =============================================================================
    time0 = time()
    pair_results, data_dict = analyze_each_pair_of_folders(folders, **settings)

    def parts_match(pr):
        # Y5 matches with nothing
        if 'Y5' in pr['name']:
            return False
        # Ls and other Ys match each other
        elif ('L' in pr['name'][0] or 'Y' in pr['name'][0]) and ('L' in pr['name'][1] or 'Y' in pr['name'][1]):
            return True
        # Others match if they share a letter and are both <=3 or both >=4
        elif (pr['name'][0][0] == pr['name'][1][0]
              and ((int(pr['name'][0][1]) <= 3 and int(pr['name'][1][1]) <= 3)
                   or (int(pr['name'][0][1]) >= 4 and int(pr['name'][1][1]) >= 4))):
            return True
        else:
            return False

    for pr in pair_results:
        pr['same_part'] = parts_match(pr)

    for pair_result in pair_results:
        m, ux, uy, q, s = len(pair_result['matched']), len(pair_result['unmatched'][0]), len(
            pair_result['unmatched'][1]), pair_result['quality'], pair_result['stretch']
        print(f'{pair_result['name']} {m:3d} {ux:3d} {uy:3d}  {
              pair_result['match_probability']:.3f} {q:6.3f} {s:7.5f} {pair_result['same_part']}')

    if 'save' not in settings or settings['save']:
        save_tag = settings['save_tag'] if 'save_tag' in settings else ''
        with open('pair_results'+save_tag+'.pkl', 'wb') as outp:
            pickle.dump(pair_results, outp, pickle.HIGHEST_PROTOCOL)
        with open('data_dict'+save_tag+'.pkl', 'wb') as outp:
            pickle.dump(data_dict, outp, pickle.HIGHEST_PROTOCOL)

    mpthresh = np.linspace(0, 1, 100)
    match_recall = np.ones_like(mpthresh)
    nomatch_recall = np.ones_like(mpthresh)
    match_precision = np.ones_like(mpthresh)
    nomatch_precision = np.ones_like(mpthresh)
    accuracy = np.ones_like(mpthresh)
    match_true = 0
    nomatch_true = 0
    for pr in pair_results:
        if pr['same_part']:
            match_true += 1
        else:
            nomatch_true += 1
    for i, mpt in enumerate(mpthresh):
        pred_match_correct = 0
        pred_match_wrong = 0
        pred_nomatch_correct = 0
        pred_nomatch_wrong = 0
        for pr in pair_results:
            if pr['match_probability'] > mpt:
                if pr['same_part']:
                    pred_match_correct += 1
                else:
                    pred_match_wrong += 1
            else:
                if pr['same_part']:
                    pred_nomatch_wrong += 1
                else:
                    pred_nomatch_correct += 1
        match_recall[i] = pred_match_correct / \
            (pred_match_correct+pred_nomatch_wrong) if pred_match_correct+pred_nomatch_wrong > 0 else np.nan
        nomatch_recall[i] = pred_nomatch_correct / \
            (pred_nomatch_correct+pred_match_wrong) if pred_nomatch_correct+pred_match_wrong > 0 else np.nan
        match_precision[i] = pred_match_correct / \
            (pred_match_correct+pred_match_wrong) if pred_match_correct+pred_match_wrong > 0 else np.nan
        nomatch_precision[i] = pred_nomatch_correct / \
            (pred_nomatch_correct+pred_nomatch_wrong) if pred_nomatch_correct+pred_nomatch_wrong > 0 else np.nan
        accuracy[i] = (pred_match_correct+pred_nomatch_correct)/(pred_match_correct+pred_nomatch_correct
                                                                 + pred_match_wrong+pred_nomatch_wrong)

    pf.line_plot(mpthresh, [match_recall, nomatch_recall, match_precision, nomatch_precision, accuracy],
                 legend=['match recall', 'nomatch recall', 'match precision', 'nomatch precision', 'accuracy'],
                 x_label='Matching Probability Threshold', legend_location=(0.02, 0.4), line_width=6, cmap='list',
                 cmap_custom=['darkblue', 'b', 'darkred', 'r', 'g', 'y', 'pink', 'y'],
                 v_line_pos=[0.1*i for i in range(10)], vlinewidth=1, y_lim=[0, 1.05],
                 x_lim=[mpthresh[0], mpthresh[-1]])
    pf.line_plot(mpthresh,
                 [match_recall, nomatch_recall, match_precision, nomatch_precision, accuracy,
                  0.957*np.ones_like(mpthresh), 0.618*np.ones_like(mpthresh), 0.957*np.ones_like(mpthresh)],
                 legend=['match recall', 'nomatch recall', 'match precision', 'nomatch precision', 'accuracy',
                         '20220328 ML Recall', '20220328 ML Precision', '20220328 ML Accuracy'],
                 x_label='Matching Probability Threshold', legend_location=(0.02, 0.3), line_width=6, cmap='list',
                 cmap_custom=['darkblue', 'b', 'darkred', 'r', 'g', 'y', 'pink', 'y'],
                 v_line_pos=[0.1*i for i in range(10)], vlinewidth=1, y_lim=[0, 1.05],
                 x_lim=[mpthresh[0], mpthresh[-1]])

    print(f'done after {time()-time0} s')


if __name__ == '__main__':
    main()
