import numpy as np
import MetroLaserLARS.plotfunctions as pf
from time import time
import pickle
from MetroLaserLARS.LarsFunctions import analyze_each_pair_of_folders
import os
from os import path as osp

np.set_printoptions(precision=3)


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


def analyze_pair_results(pair_results, data_dict, settings):
    save_results = settings['save_results'] if 'save_results' in settings else False
    save_data = settings['save_data'] if 'save_data' in settings else False
    save_tag = '_'+settings['save_tag'] if 'save_tag' in settings else ''
    save_folder = settings['save_folder']
    save_plots = settings['save_plots'] if 'save_plots' in settings else False
    show_plots = settings['show_plots'] if 'show_plots' in settings else False

    for pr in pair_results:
        pr['same_part'] = parts_match(pr)

    for pair_result in pair_results:
        m, ux, uy, q, s = len(pair_result['matched']), len(pair_result['unmatched'][0]), len(
            pair_result['unmatched'][1]), pair_result['quality'], pair_result['stretch']
        print(f'{pair_result['name']} {m:3d} {ux:3d} {uy:3d}  {
              pair_result['match_probability']:.3f} {q:6.3f} {s:7.5f} {pair_result['same_part']}')

    if save_results:
        save_path = 'pair_results'+save_tag+'.pkl' if 'save_folder' not in settings else\
            osp.join(save_folder, 'pair_results'+save_tag+'.pkl')
        with open(save_path, 'wb') as outp:
            pickle.dump(pair_results, outp, pickle.HIGHEST_PROTOCOL)
    if save_data:
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

    if 'plot_classification' in settings and settings['plot_classification']:
        pf.line_plot(mpthresh, [match_recall, nomatch_recall, match_precision, nomatch_precision, accuracy],
                     legend=['match recall', 'nomatch recall', 'match precision', 'nomatch precision', 'accuracy'],
                     x_label='Matching Probability Threshold', legend_location=(0.02, 0.4), line_width=6, cmap='list',
                     cmap_custom=['darkblue', 'b', 'darkred', 'r', 'g', 'y', 'pink', 'y'],
                     v_line_pos=[0.1*i for i in range(10)], vlinewidth=1, y_lim=[0, 1.05],
                     x_lim=[mpthresh[0], mpthresh[-1]],
                     fname=osp.join(save_folder, 'classification_stats'+save_tag+'.png') if save_plots else None,
                     show_plot_in_spyder=show_plots)
        pf.line_plot(mpthresh,
                     [match_recall, nomatch_recall, match_precision, nomatch_precision, accuracy,
                      0.957*np.ones_like(mpthresh), 0.618*np.ones_like(mpthresh), 0.957*np.ones_like(mpthresh)],
                     legend=['match recall', 'nomatch recall', 'match precision', 'nomatch precision', 'accuracy',
                             '20220328 ML Recall', '20220328 ML Precision', '20220328 ML Accuracy'],
                     x_label='Matching Probability Threshold', legend_location=(0.02, 0.3), line_width=6, cmap='list',
                     cmap_custom=['darkblue', 'b', 'darkred', 'r', 'g', 'y', 'pink', 'y'],
                     v_line_pos=[0.1*i for i in range(10)], vlinewidth=1, y_lim=[0, 1.05],
                     x_lim=[mpthresh[0], mpthresh[-1]],
                     fname=osp.join(save_folder, 'classification_comparison'+save_tag+'.png') if save_plots else None,
                     show_plot_in_spyder=show_plots)


def run_analysis(folders, settings):
    time0 = time()
    pair_results, data_dict = analyze_each_pair_of_folders(folders, **settings)
    analyze_pair_results(pair_results, data_dict, settings)
    print(f"""


Done!


All code finished running after a total of {time()-time0} s""")
    return


def get_subfolders(folder):
    subfolders = []
    for item in os.listdir(folder):
        item_path = osp.join(folder, item)
        if osp.isdir(item_path):
            subfolders.append(item_path)
    return subfolders


def LARS_Comparison_from_app(settings):
    folders = get_subfolders(settings['directory'])
    run_analysis(folders, settings)
    return


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

    settings['save_data'] = False  # Whether to save the data in a .pkl file.
    settings['save_results'] = False  # Whether to save the results in a .pkl file.
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
    folders = [r'K:\MD05_2022_JDT\Data\20240930 Adityan test\x6 646',
               r'K:\MD05_2022_JDT\Data\20240930 Adityan test\x6 655']

    # L Brackets
    folders = [r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L1',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L2',
               r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L3']

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
    # folders = [r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L1',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L2',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\L3',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\L2',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M1',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M2',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M3',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M4',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\M5',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N1',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N2',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N3',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N4',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N5',
    #            r'K:\DI02_2018_PhII_JT\Data\20210502 - N-, M-, L-type brackets - free-floating - 5 test points - 10-60 kHz\Data_modernformat\N6',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\O1',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\O2',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\O4',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\Y5',
    #            r'K:\DI02_2018_PhII_JT\Data\20210320 - L-, Y-, O-type brackets - free-floating - 5 test points - 10-60 kHz\Data original\Y7']

    # CODE
    # MODIFY WITH CARE
    # =============================================================================
    run_analysis(folders, settings)


if __name__ == '__main__':
    main()
