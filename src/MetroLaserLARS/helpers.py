# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:49:38 2024

@author: KOlson
"""
# External imports
import numpy as np
from numpy.typing import ArrayLike
from numpy.testing import assert_equal
from scipy.stats import gmean
import os.path as osp
import pickle


def are_equal(first, second):
    try:
        assert_equal(first, second)
        return True
    except AssertionError:
        return False


def can_skip_calculation(calculation: str = 'fitting', **settings: dict) -> bool:
    """
    Determines if parts of the calculation (defined by `calculation`)
    can be skipped due to loaded data and equivalent settings.

    Parameters
    ----------
    calculation : str, optional
        The default is 'fitting', which determines if fitting peaks can be skipped.
        'matching' determines if both peak fitting and peak matching can be skipped.
    **settings : TYPE
        Settings dictionary.

    Returns
    -------
    bool
        Whether the calculation can be skipped.
    """
    pickled_data_path = settings['pickled_data_path'] if 'pickled_data_path' in settings else None
    plot_detail = settings['plot_detail'] if 'plot_detail' in settings else False
    save_data = settings['save_data'] if 'save_data' in settings else False
    save_results = settings['save_results'] if 'save_results' in settings else False
    plot_recursive_noise = settings['plot_recursive_noise'] if 'plot_recursive_noise' in settings else False
    recursive_noise_reduction = settings['recursive_noise_reduction'] if 'recursive_noise_reduction' in settings else False
    peak_fitting_strategy = settings['peak_fitting_strategy'] if 'peak_fitting_strategy' in settings else 'Standard'
    standard_peak_fitting = peak_fitting_strategy == 'Standard'
    ml_peak_fitting = peak_fitting_strategy == 'Machine Learning'

    skip_calc = True

    if pickled_data_path:
        settings_path = osp.join(osp.split(pickled_data_path)[0], 'settings.pkl')
        pr_path = osp.join(osp.split(pickled_data_path)[0],
                           osp.split(pickled_data_path)[1].replace('data_dict', 'pair_results'))
        if osp.isfile(settings_path) and (osp.isfile(pr_path) or calculation not in ['matching']):
            try:
                with open(settings_path, 'rb') as f:
                    settings_saved = pickle.load(f)
            except Exception:
                skip_calc = False
                return skip_calc

            settings_to_compare = settings.copy()
            settings_to_compare.pop('status_label', None)
            diff_keys = [key for key in set(settings_to_compare.keys()).union(settings_saved.keys())
                         if settings.get(key) != settings_saved.get(key)]
            # Always need recalculation due to different data structures, outputs, or processing strategies
            skip_calc = skip_calc and not (
                'directory' in diff_keys
                or 'frange' in diff_keys
                or 'combine' in diff_keys
                or 'grouped_folders' in diff_keys
                or 'PRINT_MODE' in diff_keys
                or 'peak_fitting_strategy' in diff_keys
                or (save_data and 'save_data' in diff_keys)
                or (save_results and 'save_results' in diff_keys)
                or 'save_tag' in diff_keys
                or 'save_folder' in diff_keys
            )
            # Refit peaks if these are different, note skipping matching requires fitting to be skipped
            if calculation in ['fitting', 'matching']:
                skip_calc = skip_calc and not (
                    ('plot_detail' in diff_keys and standard_peak_fitting)
                    or ('plot_recursive_noise' in diff_keys and recursive_noise_reduction and standard_peak_fitting)
                    or ('plot' in diff_keys and (plot_detail or plot_recursive_noise) and standard_peak_fitting)
                    or (('show_plots' in diff_keys or 'save_plots' in diff_keys or 'peak_plot_width' in diff_keys)
                        and (plot_detail or (plot_recursive_noise and recursive_noise_reduction))
                        and standard_peak_fitting
                        )
                    or ('baseline_smoothness' in diff_keys and standard_peak_fitting)
                    or ('baseline_polyorder' in diff_keys and standard_peak_fitting)
                    or ('baseline_itermax' in diff_keys and standard_peak_fitting)
                    or 'sgf_applications' in diff_keys
                    or 'sgf_windowsize' in diff_keys
                    or 'sgf_polyorder' in diff_keys
                    or ('peak_height_min' in diff_keys and standard_peak_fitting)
                    or ('peak_prominence_min' in diff_keys and standard_peak_fitting)
                    or ('peak_ph_ratio_min' in diff_keys and standard_peak_fitting)
                    or ('recursive_noise_reduction' in diff_keys and standard_peak_fitting)
                    or ('max_noise_reduction_iter' in diff_keys and recursive_noise_reduction and standard_peak_fitting)
                    or ('regularization_ratio' in diff_keys and recursive_noise_reduction and standard_peak_fitting)
                    or ('ml_threshold' in diff_keys and ml_peak_fitting)
                    or ('ml_weights_path' in diff_keys and ml_peak_fitting)
                )
            # Rematch peaks if these are different
            if calculation in ['matching']:
                skip_calc = skip_calc and not (
                    'max_stretch' in diff_keys
                    or 'num_stretches' in diff_keys
                    or 'peak_match_window' in diff_keys
                    or 'matching_penalty_order' in diff_keys
                    or 'nw_normalized' in diff_keys
                )
    else:
        skip_calc = False
    return skip_calc


def group(a: ArrayLike, maxsize: int = 1) -> list[ArrayLike]:
    """
    Groups `a` into subsets with size of at most `maxsize`.
    All but the last group are guaranteed to be length `maxsize`.

    Parameters
    ----------
    a : ArrayLike
        Array to split into smaller groups.
    maxsize : int
        Maximum size of the groups. The default is 1.

    Returns
    -------
    list[ArrayLike]
        List of grouped arrays.

    """
    a = a.copy()
    return_list = []
    while np.size(a) > 0:
        first_el = a[0]
        next_array = np.array(a[0])
        a = a[1:]
        while np.size(a) > 0 and a[0] < first_el+maxsize:
            next_array = np.append(next_array, a[0])
            a = a[1:]
        return_list.append(next_array)
    return return_list


def peaks_dict_from_array(locs):
    peaks = {}
    peaks['count'] = len(locs)
    peaks['positions'] = locs
    peaks['indices'] = np.nan*np.zeros_like(locs)
    peaks['heights'] = np.nan*np.zeros_like(locs)
    peaks['widths'] = np.nan*np.zeros_like(locs)
    peaks['lefts'] = np.nan*np.zeros_like(locs)
    peaks['rights'] = np.nan*np.zeros_like(locs)
    return peaks


def names_from_pair_results(pair_results):
    names = []
    for pr in pair_results:
        if pr['names'][0] not in names:
            names.append(pr['names'][0])
        if pr['names'][1] not in names:
            names.append(pr['names'][1])
    return names


def pair_result_from_names(pair_results, name1, name2):
    for pr in pair_results:
        if ((name1 == pr['names'][0] and name2 == pr['names'][1])
                or (name1 == pr['names'][1] and name2 == pr['names'][0])):
            return pr
    return None


def array_from_pair_results(pair_results, key: str = 'stretch'):
    names = names_from_pair_results(pair_results)
    n = len(names)
    A = np.nan*np.ones((n, n))

    for i in range(n):
        for j in range(n):
            pr = pair_result_from_names(pair_results, names[i], names[j])
            if pr is not None:
                A[i, j] = 1/pr[key] if key == 'stretch' and j > i else pr[key]

    return A


def transitivity_2(A: ArrayLike = [[]]):
    M = np.array(A)

    if len(np.shape(M)) != 2 and np.shape(M)[0] != np.shape(M)[1]:
        raise ValueError("Input must be an nxn ArrayLike object")

    n = np.shape(M)[0]

    transitivity = np.nan*np.ones_like(M)

    for i in range(n):
        for j in range(i+1, n):
            if i != j:
                transitivity[i, j] = gmean([min(M[i, k]*M[k, j]/M[i, j],
                                                M[i, j]/(M[i, k]*M[k, j]))
                                            for k in range(n)
                                            if k != i and k != j])
    return gmean(transitivity.flatten(), nan_policy='omit'), np.nanmin(transitivity)


def get_line_of_data(x, y, x0, y0, x1, y1, mask, width):
    x0idx = 0 if x0 <= np.min(x) else (len(x)-1 if x0 >= np.max(x) else np.where(x >= x0)[0][0])
    x1idx = 0 if x1 <= np.min(x) else (len(x)-1 if x1 >= np.max(x) else np.where(x > x1)[0][0]-1)
    y0idx = 0 if y0 <= np.min(y) else (len(y)-1 if y0 >= np.max(y) else np.where(y >= y0)[0][0])
    y1idx = 0 if y1 <= np.min(y) else (len(y)-1 if y1 >= np.max(y) else np.where(y > y1)[0][0]-1)
    length = int(np.hypot(x1idx-x0idx, y1idx-y0idx))
    xidx, yidx = np.linspace(x0idx, x1idx, length, dtype=int), np.linspace(y0idx, y1idx, length, dtype=int)
    for j in range(width):
        mask[yidx, xidx] = True
        xidx[xidx < len(x)-1] += 1
    return mask


def get_lines_of_data(t, f, Zxx, sweep_start, sweep_end, sweep_time, width):
    t_max = np.max(t)
    mask = np.zeros_like(Zxx, dtype=bool)
    for i in range(int(t_max/sweep_time)):
        mask = get_line_of_data(t, f, sweep_time*i, sweep_start, sweep_time*(i+1), sweep_end, mask, width)
    return mask
