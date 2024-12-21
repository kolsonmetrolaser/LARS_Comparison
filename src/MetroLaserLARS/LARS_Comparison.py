# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# External imports
import numpy as np
from time import time
import pickle
from os import listdir
from os import path as osp

# Internal imports
try:
    import plotfunctions as pf
    from LarsFunctions import analyze_each_pair_of_folders
    from helpers import can_skip_calculation
except ModuleNotFoundError:
    import MetroLaserLARS.plotfunctions as pf  # type: ignore
    from MetroLaserLARS.LarsFunctions import analyze_each_pair_of_folders  # type: ignore
    from MetroLaserLARS.helpers import can_skip_calculation  # type: ignore


def parts_match(pr, **settings):
    name0, name1 = pr['names'][0], pr['names'][1]
    folder0, folder1 = pr['folders'][0], pr['folders'][1]

    part_matching_strategy = settings['part_matching_strategy'] if 'part_matching_strategy' in settings else ''
    part_matching_text = settings['part_matching_text'] if 'part_matching_text' in settings else ''
    grouped_folders = settings['grouped_folders'] if 'grouped_folders' in settings else False

    if part_matching_strategy == 'folder' and grouped_folders:
        # If the parts have the same parent folder
        # part sets are parents and all analysis is done from the grandparent or higher
        if osp.split(osp.split(folder0)[0])[1] == osp.split(osp.split(folder1)[0])[1]:
            return True
        else:
            return False
    elif part_matching_strategy == 'list' and part_matching_text != '':
        groups = [line.split(', ') for line in part_matching_text.split('\n') if line != '']
        for g in groups:
            if name0 in g and name1 in g:
                return True
        return False
    elif part_matching_strategy == 'custom' and part_matching_text != '':
        try:
            salt = str(int(np.random.rand()*1e16))
            custom_function_text = ''
            custom_function_text = 'def part_matching_function_'+salt+'(name0, name1):\n'
            for line in [ln for ln in part_matching_text.split('\n') if ln != '']:
                custom_function_text += '    '+line+'\n'
            custom_function_text += '    return result'
            exec(custom_function_text)
            return eval('part_matching_function_'+salt+'(name0, name1)')
        except Exception as e:
            import traceback
            print('Error in custom part matching function, assuming all parts are unique...')
            print('Error:', e)
            print(traceback.format_exc())
            return False

    return False


def analyze_pair_results(pair_results, data_dict, settings):
    save_results = settings['save_results'] if 'save_results' in settings else False
    save_data = settings['save_data'] if 'save_data' in settings else False
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings and settings['save_tag'] != '' else ''
    save_folder = settings['save_folder']
    save_plots = settings['save_plots'] if 'save_plots' in settings else False
    show_plots = settings['show_plots'] if 'show_plots' in settings else False
    PRINT_MODE = settings['PRINT_MODE'] if 'PRINT_MODE' in settings else 'sparse'
    plot = settings['plot'] if 'plot' in settings else False
    plot_classification = settings['plot_classification'] if 'plot_classification' in settings else False
    slc_limits = settings['slc_limits'] if 'slc_limits' in settings else (10000, 60000)

    for pr in pair_results:
        pr['same_part'] = parts_match(pr, **settings)

    for pair_result in pair_results:
        m, ux, uy, q, s = len(pair_result['matched']), len(pair_result['unmatched'][0]), len(
            pair_result['unmatched'][1]), pair_result['quality'], pair_result['stretch']
        if PRINT_MODE in ['sparse', 'full']:
            prnames, prmp, prsp = pair_result['names'], pair_result['match_probability'], pair_result['same_part']
            print(f'{prnames} {m:3d} {ux:3d} {uy:3d}  {prmp:.3f} {q:6.3f} {s:7.5f} {prsp}')

    if save_results:
        if PRINT_MODE in ['sparse', 'full']:
            print('pickling results...')
        save_path = 'pair_results'+save_tag+'.pkl' if 'save_folder' not in settings else\
            osp.join(save_folder, 'pair_results'+save_tag+'.pkl')
        with open(save_path, 'wb') as outp:
            pickle.dump(pair_results, outp, pickle.HIGHEST_PROTOCOL)
    if save_data:
        if PRINT_MODE in ['sparse', 'full']:
            print('pickling data...')
        save_path = 'data_dict'+save_tag+'.pkl' if 'save_folder' not in settings else\
            osp.join(save_folder, 'data_dict'+save_tag+'.pkl')
        with open(save_path, 'wb') as outp:
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

    if plot and plot_classification:
        pf.line_plot(mpthresh, [match_recall, nomatch_recall, match_precision, nomatch_precision, accuracy],
                     legend=['match recall', 'nomatch recall', 'match precision', 'nomatch precision', 'accuracy'],
                     x_label='Matching Probability Threshold', legend_location=(0.02, 0.4), line_width=6, cmap='list',
                     cmap_custom=['darkblue', 'b', 'darkred', 'r', 'g', 'y', 'pink', 'y'],
                     v_line_pos=[0.1*i for i in range(10)], v_line_width=1, y_lim=[0, 1.05],
                     x_lim=[mpthresh[0], mpthresh[-1]],
                     fname=osp.join(save_folder, 'classification_stats'+save_tag) if save_plots else None,
                     show_plot_in_spyder=show_plots)
        pf.line_plot(mpthresh,
                     [match_recall, nomatch_recall, match_precision, nomatch_precision, accuracy,
                      0.957*np.ones_like(mpthresh), 0.618*np.ones_like(mpthresh), 0.957*np.ones_like(mpthresh)],
                     legend=['match recall', 'nomatch recall', 'match precision', 'nomatch precision', 'accuracy',
                             '20220328 ML Recall', '20220328 ML Precision', '20220328 ML Accuracy'],
                     x_label='Matching Probability Threshold', legend_location=(0.02, 0.3), line_width=6, cmap='list',
                     cmap_custom=['darkblue', 'b', 'darkred', 'r', 'g', 'y', 'pink', 'y'],
                     v_line_pos=[0.1*i for i in range(10)], v_line_width=1, y_lim=[0, 1.05],
                     x_lim=[mpthresh[0], mpthresh[-1]],
                     fname=osp.join(save_folder, 'classification_comparison'+save_tag) if save_plots else None,
                     show_plot_in_spyder=show_plots)
    if plot:
        vels = []
        freqs = []
        names = []
        for k in data_dict:
            ld = data_dict[k]
            vels.append(ld.newvel)
            slc = np.logical_and(ld.freq > slc_limits[0], ld.freq < slc_limits[1])
            freqs.append(ld.freq[slc]/1000)
            names.append(ld.name)
        for i, vel in enumerate(vels):
            vels[i] = .95*vel/np.max(vel)+i
        fname = 'all_spectra'
        pf.line_plot(freqs[::-1], vels[::-1], line_width=4, y_lim=(-.05, len(vels)),
                     x_lim=(slc_limits[0]/1000, slc_limits[1]/1000),
                     legend=names[::-1], legend_location='best', title='All spectra',
                     fname=osp.join(save_folder, fname) if save_plots else None,
                     fig_size=(12, 5*len(vels)), show_plot_in_spyder=show_plots,
                     x_label='Frequency (kHz)', y_label='Intensity (arb.)', y_ticks=[])


def run_analysis(folders, settings):
    time0 = time()
    np.set_printoptions(precision=4)

    save_settings = settings['save_settings'] if 'save_settings' in settings else False
    save_folder = settings['save_folder']
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings and settings['save_tag'] != '' else ''
    PRINT_MODE = settings['PRINT_MODE'] if 'PRINT_MODE' in settings else 'sparse'

    skip_fit_and_match = can_skip_calculation('matching', **settings)

    pickled_data_path = settings['pickled_data_path'] if 'pickled_data_path' in settings else None
    pr_path = osp.join(osp.split(pickled_data_path)[0],
                       osp.split(pickled_data_path)[1].replace('data_dict', 'pair_results'))
    if skip_fit_and_match:
        try:
            with open(pr_path, 'rb') as f:
                pair_results = pickle.load(f)
            with open(pickled_data_path, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception:
            skip_fit_and_match = False

    if not skip_fit_and_match:
        peak_fitting_strategy = 'Standard' if 'peak_fitting_strategy' not in settings else settings['peak_fitting_strategy']
        if peak_fitting_strategy == 'Machine Learning':
            try:
                import ml_functions as ml
            except ModuleNotFoundError:
                import MetroLaserLARS.ml_functions as ml  # type: ignore
            settings['model'], settings['label_encoder'] = ml.load_model(**settings)
        pair_results, data_dict = analyze_each_pair_of_folders(folders, **settings)
    else:
        print('skipped peak fitting and matching')

    analyze_pair_results(pair_results, data_dict, settings)
    settings.pop('model', None)
    settings.pop('label_encoder', None)

    if save_settings:
        if PRINT_MODE in ['sparse', 'full']:
            print('pickling settings...')
        save_path = 'pair_results'+save_tag+'.pkl' if 'save_folder' not in settings else\
            osp.join(save_folder, 'settings'+save_tag+'.pkl')
        with open(save_path, 'wb') as outp:
            settings_to_save = settings.copy()
            settings_to_save.pop('status_label', None)
            settings_to_save.pop('progress_bars', None)
            pickle.dump(settings_to_save, outp, pickle.HIGHEST_PROTOCOL)
    print(f"""


Done!
All code finished running after {time()-time0:.3f} s""")
    return data_dict, pair_results


def get_subfolders(folder, grouped_folders=False):
    subfolders = []
    for item in listdir(folder):
        item_path = osp.join(folder, item)
        if osp.isdir(item_path):
            if grouped_folders:
                for item2 in listdir(item_path):
                    item_path = osp.join(folder, item, item2)
                    if osp.isdir(item_path):
                        subfolders.append(item_path)
            else:
                subfolders.append(item_path)
    return subfolders


def pare_folders(folders, settings):
    data_format = settings['data_format'] if 'data_format' in settings else 'auto'
    if data_format == 'auto':
        data_format = ['.all', '.npz', '.tdms', '.csv', '.LARSsim', '.LARSspectrum']
    else:
        data_format = [data_format]

    for folder in folders[::-1]:
        folder_has_data = False
        for item in listdir(folder):
            item_path = osp.join(folder, item)
            if osp.isfile(item_path) and osp.splitext(item_path)[1] in data_format:
                folder_has_data = True
        if not folder_has_data:
            folders.remove(folder)
    return folders


def LARS_Comparison_from_app(settings):
    try:
        grouped_folders = settings['grouped_folders'] if 'grouped_folders' in settings else False

        folders = get_subfolders(settings['directory'], grouped_folders)
        folders = pare_folders(folders, settings)
        if not folders and grouped_folders:
            print("WARNING: Ignoring grouped folders because the grouped folder structure was not recognized.")

            settings['grouped_folders'] = False
            if 'part_matching_strategy' in settings and settings['part_matching_strategy'] == 'folder':
                settings['part_matching_strategy'] = 'list'
                settings['part_matching_text'] = ''
            folders = get_subfolders(settings['directory'], False)
            folders = pare_folders(folders, settings)

        data_dict, pair_results = run_analysis(folders, settings)
    except Exception as e:
        import traceback
        return -1, (e, traceback.format_exc())

    return data_dict, pair_results


if __name__ == '__main__':
    from app import run_app
    run_app()
