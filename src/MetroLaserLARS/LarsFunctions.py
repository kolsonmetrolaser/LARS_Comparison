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
from typing import Literal, Iterable
from itertools import combinations
import pickle

# Internal imports
if __name__ == '__main__':
    import LarsDataClass
    from LarsDataClass import LarsData
    import plotfunctions as pf
    from filters import airpls, sgf
    from helpers import group
    from needlemanwunsch import find_matches
else:
    from MetroLaserLARS import LarsDataClass
    from MetroLaserLARS.LarsDataClass import LarsData
    import MetroLaserLARS.plotfunctions as pf
    from MetroLaserLARS.filters import airpls, sgf
    from MetroLaserLARS.helpers import group
    from MetroLaserLARS.needlemanwunsch import find_matches


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
                   vels_filtered, xlim=[10, 60], ylim=[0, 10], vels_peaks_removed=None, iteration=None, **settings):
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
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings and settings['save_tag'] != '' else ''
    save_folder = settings['save_folder']
    save_plots = settings['save_plots'] if 'save_plots' in settings else False
    show_plots = settings['show_plots'] if 'show_plots' in settings else False

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
        detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed, vels_filtered, **settings)

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
                           vels_filtered, vels_peaks_removed=vels_peaks_removed_for_baseline, iteration=recursive_noise_iterations, **settings)

    if plot and plot_detail and recursive_noise_reduction and not plot_recursive_noise:
        detailed_plots(folder, name, peaks, freqs, vels, vels_baseline_removed, vels_rms_norm_zeroed, vels_filtered, iteration=recursive_noise_iterations, **settings)

    newvels = vels_filtered

    if 'PRINT_MODE' in settings and settings['PRINT_MODE'] == 'full':
        peaklist = peaks['positions'][np.logical_and(peaks['positions'] > frange[0]*1000,
                                                     peaks['positions'] < frange[1]*1000)]
        print(f"""for {folder}/{name}
              peaks at:     {peaklist/1000} kHz""")

    if plot and plot_detail:
        peak_groups = group(peaks['positions'][np.logical_and(peaks['positions'] > frange[0]
                            * 1000, peaks['positions'] < frange[1]*1000)]/1000, peak_plot_width)

        pf.line_plot(freqs/1000, [vels, newvels], style='.', x_lim=frange, v_line_pos=peaks['positions']/1000,
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
        possible_formats = ['.npz', '.tdms', '.all', '.csv']
        for subdir, dirs, files in os.walk(folder):
            for file in [f for f in files if any([ext in f for ext in possible_formats])]:
                format_exists = [osp.isfile(osp.splitext(osp.join(subdir, file))[0]+fmt) for fmt in possible_formats]
                file = osp.splitext(file)[0] + [fmt for fmt, fmt_ex in zip(possible_formats, format_exists) if fmt_ex][0]
                result.append(LarsData.from_file(osp.join(subdir, file), **settings))
    else:
        for subdir, dirs, files in os.walk(folder):
            for file in [f for f in files if data_format in f]:
                result.append(LarsData.from_file(osp.join(subdir, file, **settings)))
    return result


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
        alldata = Load_LARS_data(folder, **settings)

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


def same_fit_settings(settings):
    pickled_data_path = settings['pickled_data_path'] if 'pickled_data_path' in settings else None
    directory = settings['directory']
    plot_detail = settings['plot_detail'] if 'plot_detail' in settings else False
    save_data = settings['save_data'] if 'save_data' in settings else False
    save_results = settings['save_results'] if 'save_results' in settings else False
    plot_recursive_noise = settings['plot_recursive_noise'] if 'plot_recursive_noise' in settings else False
    recursive_noise_reduction = settings['recursive_noise_reduction'] if 'recursive_noise_reduction' in settings else False

    skip_fitting = True

    if pickled_data_path:
        settings_path = osp.join(osp.split(pickled_data_path)[0], 'settings.pkl')
        pr_path = osp.join(osp.split(pickled_data_path)[0],
                           osp.split(pickled_data_path)[1].replace('data_dict', 'pair_results'))
        if osp.isfile(settings_path) and osp.isfile(pr_path):
            try:
                with open(settings_path, 'rb') as f:
                    settings_saved = pickle.load(f)
            except:
                skip_fitting = False

            settings_to_compare = settings.copy()
            settings_to_compare.pop('status_label', None)
            diff_keys = [key for key in set(settings_to_compare.keys()).union(settings_saved.keys())
                         if settings.get(key) != settings_saved.get(key)]
            skip_fitting = skip_fitting and not (
                'directory' in diff_keys
                or 'frange' in diff_keys
                or 'combine' in diff_keys
                or 'grouped_folders' in diff_keys
                or 'plot_detail' in diff_keys
                or ('plot_recursive_noise' in diff_keys and recursive_noise_reduction)
                or ('plot' in diff_keys and plot_detail)
                or ('plot' in diff_keys and plot_recursive_noise)
                or ('show_plots' in diff_keys and plot_detail)
                or ('show_plots' in diff_keys and plot_recursive_noise and recursive_noise_reduction)
                or ('save_plots' in diff_keys and plot_detail)
                or ('save_plots' in diff_keys and plot_recursive_noise and recursive_noise_reduction)
                or ('peak_plot_width' in diff_keys and plot_detail)
                or ('peak_plot_width' in diff_keys and plot_recursive_noise and recursive_noise_reduction)
                or 'PRINT_MODE' in diff_keys
                or 'baseline_smoothness' in diff_keys
                or 'baseline_polyorder' in diff_keys
                or 'baseline_itermax' in diff_keys
                or 'sgf_applications' in diff_keys
                or 'sgf_windowsize' in diff_keys
                or 'sgf_polyorder' in diff_keys
                or 'peak_height_min' in diff_keys
                or 'peak_prominence_min' in diff_keys
                or 'peak_ph_ratio_min' in diff_keys
                or 'recursive_noise_reduction' in diff_keys
                or ('max_noise_reduction_iter' in diff_keys and recursive_noise_reduction)
                or ('regularization_ratio' in diff_keys and recursive_noise_reduction)
                or (save_data and 'save_data' in diff_keys)
                or (save_results and 'save_results' in diff_keys)
                or 'save_tag' in diff_keys
                or 'save_folder' in diff_keys)
    else:
        skip_fitting = False
    return skip_fitting


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

    frange = settings['frange'] if 'frange' in settings else (10, 60)
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
        if previously_analyzed_data[i] is None:
            if PRINT_MODE in ['sparse', 'full']:
                print(f'Loading and analyzing data from {f}')
            (peaks, freq, vel, newvel, name), data = LARS_analysis(folder=f, **settings)
        else:
            if (hasattr(previously_analyzed_data[i], 'analyzed_this_session')
                    and previously_analyzed_data[i].analyzed_this_session) or same_fit_settings(settings):
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
    if plot and plot_detail:
        xlims = [[frange[0], frange[1]]] +\
            [[min(frange[0]+(i-1)*peak_plot_width, frange[1]-peak_plot_width),
              min(frange[0]+(i)*peak_plot_width, frange[1])]
             for i in range(int(np.ceil((frange[1]-frange[0])/peak_plot_width)))]

        # ([10, 60], [10, 10+50/3], [10+50/3, 10+100/3], [10+100/3, 10+150/3])
        # for xlim in xlims:
        #     pf.line_plot([freqs[0]/1000, freqs[1]/1000], [newvels[0], newvels[1]], style='.',
        #                  x_lim=xlim, v_line_pos=[positions[0]/1000, positions[1]/1000],
        #                  v_line_color=['C0', 'C1'], v_line_width=[2, 2], y_norm='each', title='Unstretched peak fits')
        # for xlim in xlims:
        #     pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [newvels[0], newvels[1]], style='.',
        #                  x_lim=xlim, v_line_pos=[positions[0]/1000, positions[1]/1000*best_stretch],
        #                  v_line_color=['C0', 'C1'], v_line_width=[2, 2], y_norm='each', title='Stretched peak fits')
        # for xlim in xlims:
        #     pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [newvels[0], newvels[1]], style='.',
        #                  x_lim=xlim, v_line_pos=[matched, unmatched_X, unmatched_Y], v_line_color=['k', 'C0', 'C1'],
        #                  v_line_width=[4, 2, 2], y_norm='each', title='Stretched peak matches filtered')
        for plotnum, xlim in enumerate(xlims):
            fname = f'{names[0]} and {names[1]} Stretched peak matches raw_{plotnum+1}'+save_tag if plotnum != 0 else\
                f'{names[0]} and {names[1]} Stretched peak matches raw'
            pf.line_plot([freqs[0]/1000, freqs[1]/1000*best_stretch], [vels[0], vels[1]], style='.', x_lim=xlim,
                         v_line_pos=[matched, unmatched_X, unmatched_Y], v_line_color=['k', 'C0', 'C1'],
                         v_line_width=[4, 2, 2], y_norm='each', title='Stretched peak matches raw',
                         fname=osp.join(save_folder, fname) if save_plots else None,
                         show_plot_in_spyder=show_plots)

    matching_analysis = {'stretch': best_stretch, 'quality': best_quality, 'names': names, 'matched': matched,
                         'unmatched': [unmatched_X, unmatched_Y],
                         'match_probability': 2*len(matched)/(2*len(matched)+len(unmatched_X)+len(unmatched_Y)),
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
            print(f"""Analyzing    {osp.split(fpair[0])[1]}    and    {
                  osp.split(fpair[1])[1]}    (pair {i+1} of {len(folder_pairs)})""")
        if 'status_label' in settings:
            import tkinter
            label_text = f'Analyzing pair {i+1} of {len(folder_pairs)}'
            settings['status_label'].config(text=label_text, state=tkinter.NORMAL)
            settings['status_label'].update()
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
