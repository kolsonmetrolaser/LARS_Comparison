# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:55:54 2024

@author: KOlson
"""

directory = """
Choose STUDY, which has the file structure below.
Saved data, results, and plots will be created in Study.

STUDY
├── PART1
│   ├── measurement1
│   ├── measurement2
│   ├── ...
├── PART2
│   ├── measurement1
│   ├── ...
├── ...
├── data_dict.pkl
├── pair_results.pkl
└── plots.png

or, if "Use grouped folder structure" is set to True,
STUDY
├── GROUP1
│   ├── PART1.1
│   │   ├── measurement1
│   │   ├── measurement2
│   │   ├── ...
│   ├── PART1.2
│   │   ├── measurement1
│   │   ├── ...
│   ├── ...
├── GROUP2
│   ├── PART2.1
│   │   ├── measurement1
│   │   ├── ...
│   ├── PART2.2
│   │   ├── measurement1
│   │   ├── ...
│   ├── ...
├── ...
├── data_dict.pkl
├── pair_results.pkl
└── plots.png
"""

data_format = """
Choose a data format to load. "auto" will choose the fastest
available format while assuming that files with the same name
refer to the same measurement (e.g., p1.tdms and p1.npz are
assumed to be identical, and thus the slower one will be skipped).
"""

new_data_format = """
Choose a data format to save a copy of the raw data to. Files
will be saved in the same location as the original.
"both" will save copies in each other format.
"""

pickled_data_path = """
Choose a previously saved .pkl data file. Speeds up analysis
when analyzing the same data with different settings.
The file data_dict.pkl is created if the setting
"Save data to .pkl file" is set to True.
"""

slc_limits = """
Select the maximum and minimum frequencies (in Hz) to be
considered during data analysis. This does NOT need to
match the frequencies available in the input data.
"""

combine = """
How to combine multiple measurements associated with one part.
If set to "max", the maximum amplitude at each frequency is
used. If set to "mean", the average amplitude is used instead.
"""

plot = """
Must be set to True for any plots to be created.
"""

plot_detail = """
If True, plots showing detailed steps of the fitting process
will be created. Useful for debugging and dialing in settings.
"""

plot_recursive_noise = """
If True, plots will be created for each recursive noise
reduction step, instead of only for the final results.
Requires "Recursively reduce noise" to also be set to True.
"""

plot_classification = """
If True, plots showing statistics related to the predictive
power of peak matching thresholds will be shown.
"""

show_plots = """
If True, plots will be shown as the program runs. Depending
how the code was run, this may cause pop-up windows that show
plots and pause the analysis, or plots may be collected by
another program (e.g., the "Plots" window in Spyder).
Has no effect if "Create plots" is False.
"""

save_plots = """
If True, plots will be saved. To choose where, see "Saving".
Has no effect if "Create plots" is False.
"""

peak_plot_width = """
Choose the width of plots that show peak fits in kHz.
Has no effect if "Create plots" is False.
"""

PRINT_MODE = """
Level of detail in console print outputs. "full" is useful for
debugging. "sparse" gives a sense of calculation progress.
"""

peak_fitting_strategy = """
Algorithm to use for peak fitting. "Standard" uses a typical
peak fitting strategy with settings for a threshold as a
multiple of the noise level. "Machine Learning" uses a trained
model to fit peaks and is generally less dependent on settings.
"""

baseline_smoothness = """
Smoothing parameter for airpls baseline calculation.
"""

baseline_polyorder = """
Polynomial order parameter for airpls baseline calculation.
"""

baseline_itermax = """
Maximum iterations for airpls baseline calculation.
"""

sgf_applications = """
Number of applications of Sovitzky-Golay filter.
"""

sgf_windowsize = """
Window size, in number of data points (typically 0.5 Hz increments)
for Sovitzky-Golay filter.
"""

sgf_polyorder = """
Polynomial order for Sovitzky-Golay filter.
"""

peak_height_min = """
Minimum height to be considered a peak, expressed as a multiple of
the noise level. A peak's height is its largest value above the
baseline. Peaks with low height are barely distinquishable from noise.

Example:
Peak 1 has height 6 and prominence 5.
Peak 2 has height 4 and prominence 1.
Peak 3 has height 7 and prominence 7.

                                3
7|           1               /\\
6|          /\\              /   \\
5|         /  \\  2        /     \\
4|        /    \\/\\       /        \\
3|       /         \\    /           \\
2|      /           \\_/             \\
1| __/                               \\___
"""

peak_prominence_min = """
Minimum prominence to be considered a peak, expressed as a multiple
of the noise level. A peak's prominence is the smaller of prominences
in each direction, where the prominence in one direction is the
difference between its height and the smallest amplitude in the region
between the peak and the next larger peak (or the edge of the data).
Peaks with low prominence are "shoulders" of other peaks or are
small peaks in valleys between two larger peaks.

Example:
Peak 1 has height 6 and prominence 5.
Peak 2 has height 4 and prominence 1.
Peak 3 has height 7 and prominence 7.

                                3
7|           1               /\\
6|          /\\              /   \\
5|         /  \\  2        /     \\
4|        /    \\/\\       /        \\
3|       /         \\    /           \\
2|      /           \\_/             \\
1| __/                               \\___
"""

peak_ph_ratio_min = """
The minimum prominence-to-height ratio to be considered a peak. Peaks
with low prominence-to-height ratios are "shoulders" of other peaks.

Example:
Peak 1 has height 6 and prominence 5.
Peak 2 has height 4 and prominence 1.
Peak 3 has height 7 and prominence 7.

                                3
7|           1               /\\
6|          /\\              /   \\
5|         /  \\  2        /     \\
4|        /    \\/\\       /        \\
3|       /         \\    /           \\
2|      /           \\_/             \\
1| __/                               \\___
"""

recursive_noise_reduction = """
Use the recursive noise reduction algorithm. The noise level is calculated
as the RMS of all the data. If this is True, then peaks are removed to
recalculate the noise level as the RMS of the data without the peaks.
This is repeated recursively until the number of peaks fit does not change
upon recalculation of the noise level.
"""

max_noise_reduction_iter = """
Maximum number of iterations for the recursive noise reduction algorithm.
Has no effect if "Recursively reduce noise" is False.
"""

regularization_ratio = """
In each iteration of the noise reduction algorithm, the new noise is actually
(1-regularization_factor)*old_noise + regularization_factor*new_noise.
This should take a value between 0 and 1. Lower values are more stable, but
require more iterations. Higher values may introduce instabilities.
Has no effect if "Recursively reduce noise" is False.
"""

ml_threshold = """
Confidence threshold for machine learning peak finding. Sections of the
spectrum with peak confidence levels above this threshold will be marked
as a peak.
"""

ml_weights = """
Set weights for machine learning peak finding. A default set of weights
is included with the program.
"""

max_stretch = """
Maximum stretching factor allowed. The stretching factor is a scaling factor
for the frequencies of the measured data with respect to the reference data,
which can be caused by changes to the stiffness or density of parts and not
by defects.
"""

num_stretches = """
Number of stretches to attempt in each iteration of the stretch and match
algorithm. Higher values are slower, but more precise and less likely
to miss the globally optimal stretching value.

The final precision in the stretching factor is approximately
2*(Max stretching)/(stretches per iteration)
"""

stretching_iterations = """
Number of stretching iterations to perform. Higher values are slower, but
more precise.

The final precision in the stretching factor is approximately
2*max_stretch/(stretches_per_iteration*stretch_factor^(num_iterations-1))
"""

stretch_iteration_factor = """
Each stretching iteration reduces the search space by this factor. Larger
values are more precise, but are more likely to miss the globally optimum
stretching value.

The final precision in the stretching factor is approximately
2*max_stretch/(stretches_per_iteration*stretch_factor^(num_iterations-1))
"""

peak_match_window = """
Maximum frequency difference between peaks to be considered matching.
"""

matching_penalty_order = """
Peak matching penalties in the modified Needleman-Wunsch algorithm are calcluated
as the difference between the peaks to the power "matching penalty order". Larger
values penalize looser matches more heavily compared to very close matches, thus
favoring many medium-quality matches as opposed to some very good matches and some
poor matches.
"""

nw_normalized = """
If True, the maximum peak match distance is defined as a fractional difference
rather than an absolute frequency. That is, looser matches are allowed for
peaks at higher frequencies.

Set to True if, for example, 10000 and 10099 should match but 60000 and 60101
should not. Set to False if 10000-10099 and 60000-60599 should both match.
These examples correspond to maximum peak match distances of 100 Hz and
0.01 * peak_position / Hz, respectively.
"""

save_data = """
If True, saves the raw data to a .pkl file. Selecting the saved file in
"Pickled Data" speeds up future calculations that use the same data, for example
when calculations with different settings are being compared.
Will also save settings.pkl to speed up future calculations with similar settings.
"""

save_results = """
If True, saves the calculations results to a .pkl file. This file may be loaded
by outside programs to further analyze the results.
"""

save_tag = """
An optional addition to the .pkl files to avoid overwriting files.
"""

save_folder = """
Folder to save .pkl and plot files. By default, this is the same as the main
"Directory" under "Load Data".
"""

save_settings = """
Saving settings can make future analysis faster if peak fitting and/or
peak matching can be skipped.
"""

grouped_folders = """
Recursively explore folders. Set to True when data is separated into
multiple folder levels rather than each individual part being in a single
folder.
"""

part_matching_strategy = """
folder: Each set of equivalent parts is grouped into its own folders. Requires
"Use Grouped Folder Structure" to be set to True.
list: Define equivalent parts by listing each group on a new line in the text box.
custom: Define equivalent parts using a Python function, given the part names.
"""

reference = """
Enter a part (folder name) to be used as a reference (e.g.: Part1), or a list
of parts to compare against separated by a comma and a space
(e.g.: Part1, Part2, Part3)
"""

stretching_transitivity = """
This is an approximate measure of the quality of the stretching data. One should
expect the stretching factor between the spectra for A and C to be the same as
the stretching factor between A and B multiplied by that between B and C. That is,
stretching should be transitive (A->C = A->B->C). This number is the geometric mean
of the ratio between paths of length 1 and length 2, and should be very close to 1.
"""
