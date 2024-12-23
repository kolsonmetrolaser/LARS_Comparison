# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:56:56 2024

@author: KOlson
"""
# External imports
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.signal as sig
import os
import os.path as osp
import pathlib
from typing import Iterable
from itertools import combinations
import pickle

# Internal imports
try:
    import LarsDataClass
    from LarsDataClass import LarsData
    import plotfunctions as pf
    from filters import airpls, sgf
    from helpers import group, can_skip_calculation, peaks_dict_from_array
    from needlemanwunsch import find_matches
except ModuleNotFoundError:
    from MetroLaserLARS import LarsDataClass  # type: ignore
    from MetroLaserLARS.LarsDataClass import LarsData  # type: ignore
    import MetroLaserLARS.plotfunctions as pf  # type: ignore
    from MetroLaserLARS.filters import airpls, sgf  # type: ignore
    from MetroLaserLARS.helpers import group, can_skip_calculation, peaks_dict_from_array  # type: ignore
    from MetroLaserLARS.needlemanwunsch import find_matches  # type: ignore


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


def remove_noise(y: ArrayLike, normalize: bool = True, noise: None | ArrayLike = None)\
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
    # noise = np.std(y) if noise is None else noise
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
    if d['count']:
        d['positions'], d['heights'], d['widths'] =\
            x[peaks[0][d['indices']]], peaks[1]['peak_heights'][d['indices']], peak_widths[0][d['indices']]*(x[1]-x[0])
        d['rights'] = d['positions'] + d['widths']/2
        d['lefts'] = d['positions']-d['widths']/2
    else:
        for k in ['positions', 'heights', 'widths', 'rights', 'lefts']:
            d[k] = np.array([])
    return d


def detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed,
                   vels_filtered, xlim=[10, 60], ylim=[0, 10], vels_peaks_removed=None, iteration=None,
                   vels_peaks_removed_baseline_removed=None, **settings):
    """
    Makes optional detailed plots inside `analyze_data()`.
    """
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings and settings['save_tag'] != '' else ''
    save_folder = settings['save_folder']
    save_plots = settings['save_plots'] if 'save_plots' in settings else False
    show_plots = settings['show_plots'] if 'show_plots' in settings else False

    save_tag = f"_{iteration}"+save_tag if iteration is not None else save_tag

    kwargs = {'x_label': 'Frequency (kHz)', 'v_line_width': 1}
    pf.line_plot(freqs/1000, [vels], style='.', x_lim=xlim,
                 title=f'{folder}{name} raw data', y_lim=[-2, 200], **kwargs, y_label='Amplitude (μm/s)',
                 fname=osp.join(save_folder, f'{folder}{name} raw data'+save_tag) if save_plots else None,
                 show_plot_in_spyder=show_plots)
    pf.line_plot(freqs/1000, [vels, vels-vels_baseline_removed], style='.', x_lim=xlim,
                 title=f'{folder}{name} raw data and baseline', y_lim=[-2, 200], **kwargs, y_label='Amplitude (μm/s)',
                 fname=osp.join(save_folder, f'{folder}{name} raw data and baseline'+save_tag) if save_plots else None,
                 show_plot_in_spyder=show_plots)
    if vels_peaks_removed is not None:
        pf.line_plot(freqs/1000, [vels_peaks_removed, vels-vels_baseline_removed], style='.', x_lim=xlim,
                     title=f'{folder}{name} peaks removed and baseline', y_lim=[-2, 70],
                     **kwargs, y_label='Amplitude (μm/s)',
                     fname=osp.join(save_folder, f'{folder}{name} peaks removed and baseline'+save_tag) if save_plots else None,
                     show_plot_in_spyder=show_plots)
#     if vels_peaks_removed_baseline_removed is not None:
#         import scipy.stats as st
#         import matplotlib.pyplot as plt
#         from scipy.optimize import leastsq
#         y = vels_peaks_removed_baseline_removed.copy()
#         rms = np.sqrt(np.mean(vels_peaks_removed_baseline_removed**2))
#         stdev = np.std(vels_peaks_removed_baseline_removed)
#         mean = np.mean(vels_peaks_removed_baseline_removed)
#         print(f"""mean: {mean}    rms: {rms}    stdev: {stdev}
# ratio: {rms/stdev}""")
#         num_bins = 200
#         span = np.linspace(-rms, rms*4, num_bins)
#         h = plt.hist(y, color='gray', bins=span, density=True)
#         x = np.linspace(span[0], span[-1], 1000)
#         # plt.plot(x, st.norm.pdf(x, noise/(2), noise/2))
#         # plt.plot(x, st.halfnorm.pdf(x, 0, rms), '-r')
#         # plt.plot(x, st.halfnorm.pdf(x, 0, stdev), '-b')
#         # plt.plot(x, st.foldnorm.pdf(x, mean, 0, stdev), '-g')
#         plt.plot(x, st.norm.pdf(x, mean, stdev), '-y')

#         hx = (h[1][:-1]+h[1][1:])/2
#         hy = h[0]
#         plt.plot(hx, hy, ':k')
#         fitfunc = lambda p, x: 1/np.sqrt(2*np.pi*p[1]**2)*np.exp(-0.5*((x-p[0])/p[1])**2)
#         fitdifffunc = lambda p, x, y: y-fitfunc(p, x)
#         init = [mean, stdev]
#         out = leastsq(fitdifffunc, init, args=(hx, hy))
#         print(out[0])
#         plt.plot(hx, fitfunc(out[0], hx), '-r')

#         plt.title(f'noise histogram iteration {iteration}')
#         plt.xlim(span[0], span[-1])
#         plt.show()
    pf.line_plot(freqs/1000, [vels_baseline_removed], style='.', x_lim=xlim,
                 title=f'{folder}{name} baseline removed', y_lim=[-2, 200], **kwargs, y_label='Amplitude (μm/s)',
                 fname=osp.join(save_folder, f'{folder}{name} baseline removed'+save_tag) if save_plots else None,
                 show_plot_in_spyder=show_plots)
    pf.line_plot(freqs/1000, [vels_rms_norm_zeroed], style='.', x_lim=xlim,
                 title=f'{folder}{name} rms_norm_zero', y_lim=[0, 15], **kwargs, y_label='Amplitude (arb.)',
                 fname=osp.join(save_folder, f'{folder}{name} rms_norm_zero'+save_tag) if save_plots else None,
                 show_plot_in_spyder=show_plots)
    pf.line_plot(freqs/1000, [vels_filtered], style='.', x_lim=xlim,
                 title=f'{folder}{name} filtered data', y_lim=ylim, **kwargs, y_label='Amplitude (arb.)',
                 fname=osp.join(save_folder, f'{folder}{name} filtered data'+save_tag) if save_plots else None,
                 show_plot_in_spyder=show_plots)
    pf.line_plot(freqs/1000, [vels_filtered], style='.', x_lim=xlim, v_line_pos=peaks['positions']/1000,
                 title=f'{folder}{name} peak fits', y_lim=ylim, **kwargs, y_label='Amplitude (arb.)',
                 fname=osp.join(save_folder, f'{folder}{name} peak fits'+save_tag) if save_plots else None,
                 show_plot_in_spyder=show_plots)
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
    slc_limits = settings['slc_limits'] if 'slc_limits' in settings else (10000, 60000)
    plot = settings['plot'] if 'plot' in settings else False
    plot_detail = settings['plot_detail'] if 'plot_detail' in settings else True
    plot_recursive_noise = settings['plot_recursive_noise'] if 'plot_recursive_noise' in settings else True
    recursive_noise_reduction = settings['recursive_noise_reduction'] if\
        'recursive_noise_reduction' in settings else True
    max_noise_reduction_iter = settings['max_noise_reduction_iter'] if 'max_noise_reduction_iter' in settings else 10
    sgf_applications = settings['sgf_applications'] if 'sgf_applications' in settings else 2
    sgf_windowsize = settings['sgf_windowsize'] if 'sgf_windowsize' in settings else 101
    sgf_polyorder = settings['sgf_polyorder'] if 'sgf_polyorder' in settings else 0
    peak_plot_width = settings['peak_plot_width'] if 'peak_plot_width' in settings else 10
    regularization_ratio = settings['regularization_ratio'] if 'regularization_ratio' in settings else 0.5
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings and settings['save_tag'] != '' else ''
    save_folder = settings['save_folder']
    save_plots = settings['save_plots'] if 'save_plots' in settings else False
    show_plots = settings['show_plots'] if 'show_plots' in settings else False
    PRINT_MODE = settings['PRINT_MODE'] if 'PRINT_MODE' in settings else 'none'

    freqs = data.freq
    slc = np.logical_and(freqs > slc_limits[0], freqs < slc_limits[1])
    freqs = freqs[slc]
    folder = pathlib.Path(data.path).parts[-2]
    name = data.name

    if PRINT_MODE == 'full':
        print(f'For {folder}/{name}:')

    if len(data.freq) > 0 and len(data.vel) == 0:  # has freqs but not vels, it is a simulation file with the peaks listed in freqs
        peaks = peaks_dict_from_array(freqs)
        if PRINT_MODE == 'full':
            peaklist = peaks['positions'][np.logical_and(peaks['positions'] > slc_limits[0],
                                                         peaks['positions'] < slc_limits[1])]
            print(f"""For {folder}/{name}:
              simulated peaks at:     {peaklist/1000} kHz""")
        return peaks, freqs, data.vel, data.vel, data.name

    vels = data.vel[slc]

    vels_baseline_removed, baseline = remove_baseline(vels, **settings)

    vels_rms_norm_zeroed, noise = remove_noise(vels_baseline_removed)

    vels_filtered = sgf(vels_rms_norm_zeroed, n=sgf_applications, w=sgf_windowsize, p=sgf_polyorder)

    peaks = fit_peaks(freqs, vels_filtered, **settings)

    if plot and ((plot_detail and not recursive_noise_reduction)
                 or (recursive_noise_reduction and plot_recursive_noise)):
        detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed,
                       vels_rms_norm_zeroed, vels_filtered, **settings)

    if PRINT_MODE == 'full':
        if recursive_noise_reduction:
            print('Starting recursive noise reduction...')
        else:
            print(f'Found {peaks["count"]} peaks without noise reduction.')

    recursive_noise_iterations = 0
    while recursive_noise_reduction:
        noise_prev = noise
        vels_peaks_removed = vels.copy()
        vels_peaks_removed_for_baseline = vels.copy()
        for left, right in zip(peaks['lefts'], peaks['rights']):
            vels_peaks_removed[np.logical_and(freqs > left, freqs < right)] = np.nan
            vels_peaks_removed_for_baseline[np.logical_and(
                freqs > left, freqs < right)] = baseline[np.logical_and(freqs > left, freqs < right)]
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
        # rms = np.std(vels_peaks_removed_baseline_removed)

        vels_baseline_removed = vels.copy() - updated_baseline

        noise = regularization_ratio*rms + (1-regularization_ratio)*noise_prev

        vels_rms_norm_zeroed, noise = remove_noise(vels_baseline_removed, noise=noise)

        vels_filtered = sgf(vels_rms_norm_zeroed, n=sgf_applications, w=sgf_windowsize, p=sgf_polyorder)

        peaks_updated = fit_peaks(freqs, vels_filtered, **settings)

        if PRINT_MODE == 'full':
            pc, puc = peaks['count'], peaks_updated['count']
            print(f'{pc} peaks to {puc} peaks')

        recursive_noise_iterations += 1
        if recursive_noise_iterations >= max_noise_reduction_iter:
            if PRINT_MODE in ['sparse', 'full']:
                print('Reached maximum recursive noise iterations for', name)
            recursive_noise_reduction = False
        if peaks['count'] == peaks_updated['count']:
            recursive_noise_reduction = False
            noise = rms

        peaks = peaks_updated

        if plot and plot_detail and plot_recursive_noise:
            detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed,
                           vels_filtered, vels_peaks_removed=vels_peaks_removed_for_baseline,
                           iteration=recursive_noise_iterations,
                           vels_peaks_removed_baseline_removed=vels_peaks_removed_baseline_removed,
                           **settings)

    if plot and plot_detail and recursive_noise_reduction and not plot_recursive_noise:
        detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed,
                       vels_filtered, iteration=recursive_noise_iterations, **settings)

    newvels = vels_filtered

    if PRINT_MODE == 'full':
        peaklist = peaks['positions'][np.logical_and(peaks['positions'] > slc_limits[0],
                                                     peaks['positions'] < slc_limits[1])]
        print(f"""    peaks at:     {peaklist/1000} kHz""")

    if plot and plot_detail:
        peak_groups = group(peaks['positions'][np.logical_and(peaks['positions'] > slc_limits[0],
                                                              peaks['positions'] < slc_limits[1])]/1000, peak_plot_width)

        pf.line_plot(freqs/1000, [vels, newvels], style='.', x_lim=[slc_limits[0]/1000, slc_limits[1]/1000], v_line_pos=peaks['positions']/1000,
                     v_line_width=1, title=f'{folder}{name} peak fit', y_norm='each',
                     fname=osp.join(save_folder, f'{folder}{name} peak fit'+save_tag) if save_plots else None,
                     show_plot_in_spyder=show_plots)
        for pgnum, pg in enumerate(peak_groups):
            if np.size(pg) > 1:
                avg_pos = (pg[0]+pg[-1])/2
                xl = [avg_pos-peak_plot_width/2, avg_pos+peak_plot_width/2]
            else:
                xl = [pg-peak_plot_width/2, pg+peak_plot_width/2]
            pf.line_plot(freqs/1000, [vels, newvels], style='.', x_lim=xl, v_line_pos=peaks['positions']/1000,
                         v_line_width=1, title=f'{folder}{name} peak fit', y_norm='each',
                         fname=osp.join(save_folder, f'{folder}{name} peak fit_{pgnum+1}'+save_tag) if save_plots else None,
                         show_plot_in_spyder=show_plots)

    return peaks, freqs, vels, newvels, name


def Load_LARS_data(folder: str = '', **settings):
    data_format = settings['data_format'] if 'data_format' in settings else 'auto'

    result = []
    if data_format == 'auto':
        possible_formats = ['.npz', '.tdms', '.all', '.csv', '.LARSsim', '.LARSspectrum']
        for subdir, dirs, files in os.walk(folder):
            for file in [f for f in files if any([ext in f for ext in possible_formats])]:
                format_exists = [osp.isfile(osp.splitext(osp.join(subdir, file))[0]+fmt) for fmt in possible_formats]
                file = osp.splitext(file)[0] + [fmt for fmt,
                                                fmt_ex in zip(possible_formats, format_exists) if fmt_ex][0]
                result.append(LarsData.from_file(osp.join(subdir, file), **settings))
    else:
        for subdir, dirs, files in os.walk(folder):
            for file in [f for f in files if data_format in f]:
                result.append(LarsData.from_file(osp.join(subdir, file), **settings))
    return result


def LARS_analysis(folder: str = '', previously_loaded_data: None | LarsData = None, **settings)\
        -> tuple[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike], LarsData]:
    """
    Combines data in `folder`

    Largely a wrapper for `analyze_data`.

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
    peak_fitting_strategy = 'Standard' if 'peak_fitting_strategy' not in settings else settings['peak_fitting_strategy']

    if folder == '':
        return None

    if previously_loaded_data is None:
        # Load data
        alldata = Load_LARS_data(folder, **settings)

        # Analyze data
        combine = settings['combine'] if 'combine' in settings else 'max'
        if len(alldata) > 1:
            data_to_analyze = LarsDataClass.combine(alldata, combine)
        else:
            data_to_analyze = alldata[0]
    else:
        print(f'Using previously loaded data for {previously_loaded_data.name}')
        data_to_analyze = previously_loaded_data

    if peak_fitting_strategy == 'Machine Learning':
        try:
            import ml_functions as ml
        except ModuleNotFoundError:
            import MetroLaserLARS.ml_functions as ml  # type: ignore
        analysis = ml.analyze_data(data_to_analyze, **settings)
    elif peak_fitting_strategy == 'Standard':
        analysis = analyze_data(data_to_analyze, **settings)
    else:
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

    slc_limits = settings['slc_limits'] if 'slc_limits' in settings else (10000, 60000)
    peak_plot_width = settings['peak_plot_width'] if 'peak_plot_width' in settings else 20
    max_stretch = settings['max_stretch'] if 'max_stretch' in settings else 0.02
    num_stretches = settings['num_stretches'] if 'num_stretches' in settings else 1000
    stretching_iterations = settings['stretching_iterations'] if 'stretching_iterations' in settings else 5
    stretch_iteration_factor = settings['stretch_iteration_factor'] if 'stretch_iteration_factor' in settings else 5
    matching_penalty_order = settings['matching_penalty_order'] if 'matching_penalty_order' in settings else 1
    peak_match_window = settings['peak_match_window'] if 'peak_match_window' in settings else 150
    nw_normalized = settings['nw_normalized'] if 'nw_normalized' in settings else False
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings and settings['save_tag'] != '' else ''
    save_folder = settings['save_folder']
    save_plots = settings['save_plots'] if 'save_plots' in settings else False
    show_plots = settings['show_plots'] if 'show_plots' in settings else False
    plot = settings['plot'] if 'plot' in settings else False
    plot_detail = settings['plot_detail'] if 'plot' in settings else False
    PRINT_MODE = settings['PRINT_MODE'] if 'PRINT_MODE' in settings else 'sparse'

    # collect peak positions, and the frequency, raw velocity, and smoothed velocity vectors from each folder
    positions = []
    names = []
    freqs = []
    vels = []
    newvels = []
    datas = []
    for i, f in enumerate(folders):
        if (hasattr(previously_analyzed_data[i], 'analyzed_this_session')
                and previously_analyzed_data[i].analyzed_this_session) or can_skip_calculation('fitting', **settings):
            if PRINT_MODE in ['sparse', 'full']:
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
        print(f'Best matches found with quality {best_quality:.5f} at stretch {best_stretch:.5f} ± {search_space_delta:.5f}')
        # for x, y in zip(bestrx, bestry):
        #     print(f'{x:7.1f}  {best_stretch*y:7.1f}')
        print('Peak matches:')
        print('Reference')
        for x in bestrx:
            print(f'{x:7.1f}', end='  ')
        print('')
        print('Measurement')
        for y in bestry:
            print(f'{best_stretch*y:7.1f}', end='  ')
        print('')
    unmatched_X = [x/1000 for x, y in zip(bestrx, bestry) if y == -1]
    unmatched_Y = [y/1000 for x, y in zip(bestrx, bestry) if x == -1]
    matched = [(x+y)/2/1000 for x, y in zip(bestrx, bestry) if x != -1 and y != -1]
    if len(matched) > 0:
        best_quality += peak_match_window*(len(unmatched_X)+len(unmatched_Y))/2
        best_quality /= -len(matched)
    else:
        best_quality = np.nan
    if 'PRINT_MODE' in settings and settings['PRINT_MODE'] == 'full':
        print(f'{len(unmatched_X)} unmatched peaks in reference at {np.array(unmatched_X)} kHz')
        print(f'{len(unmatched_Y)} unmatched peaks in measurement at {np.array(unmatched_Y)} kHz')
        print(f'{len(matched)} matched peaks at {np.array(matched)} kHz, with average mistmatch of {best_quality:.2f} Hz')

    # Plot results
    if plot and plot_detail:
        xlims = [[slc_limits[0]/1000, slc_limits[1]/1000]] +\
            [[min(slc_limits[0]/1000+(i-1)*peak_plot_width, slc_limits[1]/1000-peak_plot_width),
              min(slc_limits[0]/1000+(i)*peak_plot_width, slc_limits[1]/1000)]
             for i in range(int(np.ceil((slc_limits[1]-slc_limits[0])/1000/peak_plot_width)))]

        for plotnum, xlim in enumerate(xlims):
            fname = f'{names[0]} and {names[1]} Stretched peak matches raw_{plotnum+1}'+save_tag if plotnum != 0 else\
                f'{names[0]} and {names[1]} Stretched peak matches raw'
            pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [vels[0], vels[1]], style='.', x_lim=xlim,
                         v_line_pos=[matched, unmatched_X, unmatched_Y], v_line_color=['k', 'C0', 'C1'],
                         v_line_width=[4, 2, 2], y_norm='each', title='Stretched peak matches raw',
                         fname=osp.join(save_folder, fname) if save_plots else None,
                         show_plot_in_spyder=show_plots)

    if len(matched) == 0 and len(unmatched_X) == 0 and len(unmatched_Y) == 0:
        match_probability = np.nan
    else:
        match_probability = 2*len(matched)/(2*len(matched)+len(unmatched_X)+len(unmatched_Y))
    matching_analysis = {'stretch': best_stretch, 'quality': best_quality, 'names': names, 'matched': matched,
                         'unmatched': [unmatched_X, unmatched_Y],
                         'match_probability': match_probability,
                         'folders': folders}

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
        raise Exception('Attempted to compare less than two folders of data.')

    pickled_data_path = settings['pickled_data_path'] if 'pickled_data_path' in settings else ''
    reference = settings['reference'] if 'reference' in settings else ''

    results = []
    if pickled_data_path:
        try:
            print(f'Loading data from {settings["pickled_data_path"]}')
            with open(settings['pickled_data_path'], 'rb') as inp:
                data_dict = pickle.load(inp)
            settings['progress_bars'][0].set(len(data_dict)/len(folders))
        except Exception:
            print('Loading from data pickle failed, proceeding without loading data...')
            data_dict = {}
    else:
        data_dict = {}

    all_folder_pairs = list(combinations(folders, 2))
    folder_pairs = []
    if reference:
        references = reference.split(', ') if ', ' in reference else [reference]
        for fpair in all_folder_pairs:  # accept only pairs which include a desired reference
            if osp.basename(fpair[0]) in references or osp.basename(fpair[1]) in references:
                folder_pairs.append(fpair)
        if not folder_pairs:  # if none are found, proceed with full calculation
            print('WARNING: reference supplied, but does not match any parts. Proceeding with full calculation table.')
            folder_pairs = all_folder_pairs
    else:
        folder_pairs = all_folder_pairs

    for i, fpair in enumerate(folder_pairs):
        if 'PRINT_MODE' in settings and settings['PRINT_MODE'] in ['sparse', 'full']:
            print(f"Analyzing    {osp.split(fpair[0])[1]}    and    {osp.split(fpair[1])[1]}    (pair {i+1} of {len(folder_pairs)})")
        if 'status_label' in settings:
            import tkinter
            label_text = f'Analyzing pair {i+1} of {len(folder_pairs)}'
            settings['status_label'].config(text=label_text, state=tkinter.NORMAL)
            settings['status_label'].update()
            settings['progress_bars'][0].set(len(data_dict)/len(folders))
            settings['progress_bars'][1].set((i)/len(folder_pairs))
        data_0 = data_dict[fpair[0]] if fpair[0] in data_dict.keys() else None
        data_1 = data_dict[fpair[1]] if fpair[1] in data_dict.keys() else None
        matching_analysis, datas = compare_LARS_measurements(
            fpair, previously_analyzed_data=(data_0, data_1), **settings)
        results.append(matching_analysis)
        for data in datas:
            if data.path not in data_dict.keys():
                data_dict[data.path] = data
        if 'progress_bars' in settings:
            settings['progress_bars'][0].set(len(data_dict)/len(folders))
            settings['progress_bars'][1].set((i+1)/len(folder_pairs))
    for datapath in data_dict:
        data_dict[datapath].analyzed_this_session = False
    return results, data_dict


if __name__ == '__main__':
    from app import run_app
    run_app()
