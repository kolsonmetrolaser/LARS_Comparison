# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# Import the tkinter module
import tkinter as tk
from tkinter import filedialog, DoubleVar, StringVar, Label, Entry, Button, OptionMenu, IntVar


def submit():
    directory_var = directory_entry.get()
    frange_min_var = frange_min_entry.get()
    frange_max_var = frange_max_entry.get()
    combine_var = combine_menu.get()
    plot_var = True if plot_menu.get() == 'True' else False
    plot_detail_var = True if plot_detail_menu.get() == 'True' else False
    plot_recursive_noise_var = True if plot_recursive_noise_menu.get() == 'True' else False
    if directory_var:
        print(f"""
directory: {directory_var}
frange: {(frange_min_var, frange_max_var)}
combine: {combine_var}
plot (detail, recursive): {plot_var} {(plot_detail_var, plot_recursive_noise_var)}
              """)
    else:
        print("No directory entered.")


def select_directory():
    directory = filedialog.askdirectory(title="Select a Directory")
    if directory:
        directory_entry.delete(0, tk.END)  # Clear the directory_entry box
        directory_entry.insert(0, directory)  # Insert the selected directory


def select_pickled_data_path():
    pickled_data_path = filedialog.askpickled_data_path(title="Select a Folder with Pickled Data")
    if pickled_data_path:
        pickled_data_path_entry.delete(0, tk.END)  # Clear the pickled_data_path_entry box
        pickled_data_path_entry.insert(0, pickled_data_path)  # Insert the selected pickled_data_path


def update_peak_match_window_label(entry_text):
    # Update the label text with the selected option
    txt = "Hz" if entry_text == 'False' else '* peak position / Hz'
    peak_match_window_label2.config(text=txt)


def update_save_tag_label(*args):
    # Update the label text with the selected option
    txt = "Save filename: data_dict_... and peak_results" if save_tag_var.get() == '' else 'Save filename: data_dict_... and peak_results_'
    save_tag_label.config(text=txt)


# Create the main window
root = tk.Tk()
root.title("LARS Comparison Settings")


def heading(txt, frame=root, lvl=1, padding=True):
    fonts = [("Helvetica", 14, "bold"), ("Helvetica", 10, "bold")]
    h = tk.Label(frame, text=txt, font=fonts[lvl-1])
    if padding:
        h.pack(**padding_heading)
    else:
        h.pack()
    return h


# Global info
bool_options = ['True', 'False']
padding_heading = {'pady': 10, 'padx': 10}
padding_setting = {'pady': 4, 'padx': 4}
padding_option = {'pady': 0, 'padx': 4}

heading("Load Data", frame=root, padding=False)
# DIRECTORY

frame_directory = tk.Frame(root)
frame_directory.pack(**padding_setting, side=tk.TOP)
heading("Directory", lvl=2, frame=frame_directory)

directory_label = Label(frame_directory, text="Enter path to LARS data or select a folder:")

directory_entry = Entry(frame_directory, width=40)

directory_button = Button(frame_directory, text="Open", command=select_directory)
directory_var = StringVar(root)

directory_label.pack(side=tk.LEFT, padx=4)
directory_button.pack(side=tk.LEFT, padx=4)
directory_entry.pack(side=tk.LEFT, padx=4)

heading("OR", lvl=2, frame=root, padding=False)

# pickled_data_path

frame_pickled_data_path = tk.Frame(root)
frame_pickled_data_path.pack(**padding_setting, side=tk.TOP)
heading("Pickled data", lvl=2, frame=frame_pickled_data_path)

pickled_data_path_label = Label(frame_pickled_data_path, text="Enter path to pickled data or select a folder:")

pickled_data_path_entry = Entry(frame_pickled_data_path, width=40)

pickled_data_path_button = Button(frame_pickled_data_path, text="Open", command=select_pickled_data_path)
pickled_data_path_var = StringVar(root)

pickled_data_path_label.pack(side=tk.LEFT, padx=4)
pickled_data_path_button.pack(side=tk.LEFT, padx=4)
pickled_data_path_entry.pack(side=tk.LEFT, padx=4)

heading("---------- Settings ----------", frame=root)

# DATA DEFINITIONS

# frange and slc limits
frame_frange = tk.Frame(root)
frame_frange.pack(**padding_setting, side=tk.TOP)
heading("Data", frame=frame_frange)

frange_label = Label(frame_frange, text="The data minimum and maximum frequency:")
frange_label2 = Label(frame_frange, text="-")
frange_label3 = Label(frame_frange, text="Hz")

frange_min_var, frange_max_var = DoubleVar(root), DoubleVar(root)
frange_min_entry, frange_max_entry = Entry(frame_frange, width=6), Entry(frame_frange, width=6)
frange_min_entry.insert(0, 10000)
frange_max_entry.insert(0, 60000)

frange_label.pack(side=tk.LEFT)
frange_min_entry.pack(side=tk.LEFT)
frange_label2.pack(side=tk.LEFT)
frange_max_entry.pack(side=tk.LEFT)
frange_label3.pack(side=tk.LEFT)

# combine
frame_combine = tk.Frame(root)
frame_combine.pack(**padding_option, side=tk.TOP)

combine_label = Label(frame_combine, text="How data within a folder should be combined:")

combine_options = ['max', 'mean']
combine_var = StringVar(root)
combine_var.set(combine_options[0])
combine_menu = OptionMenu(frame_combine, combine_var, *combine_options)

combine_label.pack(side=tk.LEFT)
combine_menu.pack(side=tk.LEFT)

# PLOTTING AND PRINTING

# plot, plot_detail, and plot_recursive_noise
frame_plot = tk.Frame(root)
frame_plot.pack(**padding_option, side=tk.TOP)
heading("Plotting and Printing", frame=frame_plot)

plot_label = Label(frame_plot, text="Create plots (with fitting details) (of recursive noise iterations):")
plot_label.pack(side=tk.LEFT)

plot_var = StringVar(root)
plot_var.set(bool_options[1])
plot_menu = OptionMenu(frame_plot, plot_var, *bool_options)
plot_menu.pack(side=tk.LEFT)
plot_detail_var = StringVar(root)
plot_detail_var.set(bool_options[1])
plot_detail_menu = OptionMenu(frame_plot, plot_detail_var, *bool_options)
plot_detail_menu.pack(side=tk.LEFT)
plot_recursive_noise_var = StringVar(root)
plot_recursive_noise_var.set(bool_options[1])
plot_recursive_noise_menu = OptionMenu(frame_plot, plot_recursive_noise_var, *bool_options)
plot_recursive_noise_menu.pack(side=tk.LEFT)

# peak_plot_width
frame_peak_fit_plot_width = tk.Frame(root)
frame_peak_fit_plot_width.pack(**padding_setting, side=tk.TOP)

peak_plot_width_label = Label(frame_peak_fit_plot_width, text="Width of peak fit plots:")
peak_plot_width_label2 = Label(frame_peak_fit_plot_width, text="kHz")

peak_plot_width_var = DoubleVar(root)
peak_plot_width_entry = Entry(frame_peak_fit_plot_width, width=6)
peak_plot_width_entry.insert(0, 20)

peak_plot_width_label.pack(side=tk.LEFT)
peak_plot_width_entry.pack(side=tk.LEFT)
peak_plot_width_label2.pack(side=tk.LEFT)

# PRINT_MODE
frame_PRINT_MODE = tk.Frame(root)
frame_PRINT_MODE.pack(**padding_option, side=tk.TOP)

PRINT_MODE_options = ['none', 'sparse', 'full']
PRINT_MODE_label = Label(frame_PRINT_MODE, text="Print details:")

PRINT_MODE_var = StringVar(root)
PRINT_MODE_var.set(PRINT_MODE_options[1])
PRINT_MODE_menu = OptionMenu(frame_PRINT_MODE, PRINT_MODE_var, *PRINT_MODE_options)

PRINT_MODE_label.pack(side=tk.LEFT)
PRINT_MODE_menu.pack(side=tk.LEFT)


# PEAK FITTING
frame_peak_fit = tk.Frame(root)
frame_peak_fit.pack(side=tk.TOP)
heading("Peak Fitting", frame=frame_peak_fit)

frame_peak_fitl = tk.Frame(frame_peak_fit)
frame_peak_fitl.pack(side=tk.LEFT)
frame_peak_fitr = tk.Frame(frame_peak_fit)
frame_peak_fitr.pack(side=tk.LEFT)
heading("Baseline", lvl=2, frame=frame_peak_fitl, padding=False)
heading("Smoothing", lvl=2, frame=frame_peak_fitr, padding=False)

# baseline_smoothness
frame_baseline_smoothness = tk.Frame(frame_peak_fitl)
frame_baseline_smoothness.pack(**padding_setting, side=tk.TOP)

baseline_smoothness_label = Label(frame_baseline_smoothness, text="Basline smoothness: 10^")

baseline_smoothness_var = DoubleVar(root)
baseline_smoothness_entry = Entry(frame_baseline_smoothness, width=6)
baseline_smoothness_entry.insert(0, 12)

baseline_smoothness_label.pack(side=tk.LEFT)
baseline_smoothness_entry.pack(side=tk.LEFT)

# baseline_polyorder
frame_baseline_polyorder = tk.Frame(frame_peak_fitl)
frame_baseline_polyorder.pack(**padding_setting, side=tk.TOP)

baseline_polyorder_label = Label(frame_baseline_polyorder, text="Basline polyorder:")

baseline_polyorder_var = IntVar(root)
baseline_polyorder_entry = Entry(frame_baseline_polyorder, width=6)
baseline_polyorder_entry.insert(0, 2)

baseline_polyorder_label.pack(side=tk.LEFT)
baseline_polyorder_entry.pack(side=tk.LEFT)

# baseline_itermax
frame_baseline_itermax = tk.Frame(frame_peak_fitl)
frame_baseline_itermax.pack(**padding_setting, side=tk.TOP)

baseline_itermax_label = Label(frame_baseline_itermax, text="Basline itermax:")

baseline_itermax_var = IntVar(root)
baseline_itermax_entry = Entry(frame_baseline_itermax, width=6)
baseline_itermax_entry.insert(0, 10)

baseline_itermax_label.pack(side=tk.LEFT)
baseline_itermax_entry.pack(side=tk.LEFT)

# sgf_windowsize
frame_sgf_windowsize = tk.Frame(frame_peak_fitr)
frame_sgf_windowsize.pack(**padding_setting, side=tk.TOP)

sgf_windowsize_label = Label(frame_sgf_windowsize, text="SGF Window Size:")

sgf_windowsize_var = IntVar(root)
sgf_windowsize_entry = Entry(frame_sgf_windowsize, width=6)
sgf_windowsize_entry.insert(0, 101)

sgf_windowsize_label.pack(side=tk.LEFT)
sgf_windowsize_entry.pack(side=tk.LEFT)

# sgf_applications
frame_sgf_applications = tk.Frame(frame_peak_fitr)
frame_sgf_applications.pack(**padding_setting, side=tk.TOP)

sgf_applications_label = Label(frame_sgf_applications, text="SGF Applications:")

sgf_applications_var = IntVar(root)
sgf_applications_entry = Entry(frame_sgf_applications, width=6)
sgf_applications_entry.insert(0, 2)

sgf_applications_label.pack(side=tk.LEFT)
sgf_applications_entry.pack(side=tk.LEFT)

# sgf_polyorder
frame_sgf_polyorder = tk.Frame(frame_peak_fitr)
frame_sgf_polyorder.pack(**padding_setting, side=tk.TOP)

sgf_polyorder_label = Label(frame_sgf_polyorder, text="SGF polyorder:")

sgf_polyorder_var = IntVar(root)
sgf_polyorder_entry = Entry(frame_sgf_polyorder, width=6)
sgf_polyorder_entry.insert(0, 0)

sgf_polyorder_label.pack(side=tk.LEFT)
sgf_polyorder_entry.pack(side=tk.LEFT)

# headings
heading("Peak Finding", lvl=2, frame=frame_peak_fitl, padding=False)
heading("Noise Reduction", lvl=2, frame=frame_peak_fitr, padding=False)

# peak_height_min
frame_peak_fit_height_min = tk.Frame(frame_peak_fitl)
frame_peak_fit_height_min.pack(**padding_setting, side=tk.TOP)

peak_height_min_label = Label(frame_peak_fit_height_min, text="Peak height minimum: noise *")

peak_height_min_var = DoubleVar(root)
peak_height_min_entry = Entry(frame_peak_fit_height_min, width=6)
peak_height_min_entry.insert(0, 0.2)

peak_height_min_label.pack(side=tk.LEFT)
peak_height_min_entry.pack(side=tk.LEFT)

# peak_prominence_min
frame_peak_fit_prominence_min = tk.Frame(frame_peak_fitl)
frame_peak_fit_prominence_min.pack(**padding_setting, side=tk.TOP)

peak_prominence_min_label = Label(frame_peak_fit_prominence_min, text="Peak prominence minimum: noise *")

peak_prominence_min_var = DoubleVar(root)
peak_prominence_min_entry = Entry(frame_peak_fit_prominence_min, width=6)
peak_prominence_min_entry.insert(0, 0.2)

peak_prominence_min_label.pack(side=tk.LEFT)
peak_prominence_min_entry.pack(side=tk.LEFT)

# peak_ph_ratio_min
frame_peak_fit_ph_ratio_min = tk.Frame(frame_peak_fitl)
frame_peak_fit_ph_ratio_min.pack(**padding_setting, side=tk.TOP)

peak_ph_ratio_min_label = Label(frame_peak_fit_ph_ratio_min, text="Peak prominence-to-height minimum:")

peak_ph_ratio_min_var = DoubleVar(root)
peak_ph_ratio_min_entry = Entry(frame_peak_fit_ph_ratio_min, width=6)
peak_ph_ratio_min_entry.insert(0, 0.5)

peak_ph_ratio_min_label.pack(side=tk.LEFT)
peak_ph_ratio_min_entry.pack(side=tk.LEFT)

# recursive_noise_reduction
frame_recursive_noise_reduction = tk.Frame(frame_peak_fitr)
frame_recursive_noise_reduction.pack(**padding_option, side=tk.TOP)

recursive_noise_reduction_label = Label(frame_recursive_noise_reduction, text="Recursively reduce noise:")

recursive_noise_reduction_var = StringVar(root)
recursive_noise_reduction_var.set(bool_options[0])
recursive_noise_reduction_menu = OptionMenu(frame_recursive_noise_reduction, recursive_noise_reduction_var, *bool_options)

recursive_noise_reduction_label.pack(side=tk.LEFT)
recursive_noise_reduction_menu.pack(side=tk.LEFT)

# max_noise_reduction_iter
frame_max_noise_reduction_iter = tk.Frame(frame_peak_fitr)
frame_max_noise_reduction_iter.pack(**padding_setting, side=tk.TOP)

max_noise_reduction_iter_label = Label(frame_max_noise_reduction_iter, text="Max noise reduction iterations:")

max_noise_reduction_iter_var = IntVar(root)
max_noise_reduction_iter_entry = Entry(frame_max_noise_reduction_iter, width=6)
max_noise_reduction_iter_entry.insert(0, 10)

max_noise_reduction_iter_label.pack(side=tk.LEFT)
max_noise_reduction_iter_entry.pack(side=tk.LEFT)

# regularization_ratio
frame_regularization_ratio = tk.Frame(frame_peak_fitr)
frame_regularization_ratio.pack(**padding_setting, side=tk.TOP)

regularization_ratio_label = Label(frame_regularization_ratio, text="Noise reduction regularization factor:")

regularization_ratio_var = DoubleVar(root)
regularization_ratio_entry = Entry(frame_regularization_ratio, width=6)
regularization_ratio_entry.insert(0, 0.5)

regularization_ratio_label.pack(side=tk.LEFT)
regularization_ratio_entry.pack(side=tk.LEFT)

# PEAK MATCHING
frame_peak_match = tk.Frame(root)
frame_peak_match.pack(side=tk.TOP)
heading("Peak Matching", frame=frame_peak_match)

frame_peak_matchl = tk.Frame(frame_peak_match)
frame_peak_matchl.pack(side=tk.LEFT)
frame_peak_matchr = tk.Frame(frame_peak_match)
frame_peak_matchr.pack(side=tk.LEFT)
heading("Stretching", lvl=2, frame=frame_peak_matchl, padding=False)
heading("Matching", lvl=2, frame=frame_peak_matchr, padding=False)

# max_stretch
frame_max_stretch = tk.Frame(frame_peak_matchl)
frame_max_stretch.pack(**padding_setting, side=tk.TOP)

max_stretch_label = Label(frame_max_stretch, text="Max Stretching: 1 ±")

max_stretch_var = DoubleVar(root)
max_stretch_entry = Entry(frame_max_stretch, width=6)
max_stretch_entry.insert(0, 0.02)

max_stretch_label.pack(side=tk.LEFT)
max_stretch_entry.pack(side=tk.LEFT)

# num_stretches
frame_num_stretches = tk.Frame(frame_peak_matchl)
frame_num_stretches.pack(**padding_setting, side=tk.TOP)

num_stretches_label = Label(frame_num_stretches, text="Number of stretches per iteration:")

num_stretches_var = IntVar(root)
num_stretches_entry = Entry(frame_num_stretches, width=6)
num_stretches_entry.insert(0, 1000)

num_stretches_label.pack(side=tk.LEFT)
num_stretches_entry.pack(side=tk.LEFT)

# stretching_iterations
frame_stretching_iterations = tk.Frame(frame_peak_matchl)
frame_stretching_iterations.pack(**padding_setting, side=tk.TOP)

stretching_iterations_label = Label(frame_stretching_iterations, text="Number of stretching iterations:")

stretching_iterations_var = IntVar(root)
stretching_iterations_entry = Entry(frame_stretching_iterations, width=6)
stretching_iterations_entry.insert(0, 10)

stretching_iterations_label.pack(side=tk.LEFT)
stretching_iterations_entry.pack(side=tk.LEFT)

# stretch_iteration_factor
frame_stretch_iteration_factor = tk.Frame(frame_peak_matchl)
frame_stretch_iteration_factor.pack(**padding_setting, side=tk.TOP)

stretch_iteration_factor_label = Label(frame_stretch_iteration_factor, text="Factor to reduce stretch space each iteration:")

stretch_iteration_factor_var = DoubleVar(root)
stretch_iteration_factor_entry = Entry(frame_stretch_iteration_factor, width=6)
stretch_iteration_factor_entry.insert(0, 5)

stretch_iteration_factor_label.pack(side=tk.LEFT)
stretch_iteration_factor_entry.pack(side=tk.LEFT)

# nw_normalized
frame_nw_normalized = tk.Frame(frame_peak_matchr)
frame_nw_normalized.pack(**padding_option, side=tk.TOP)

nw_normalized_label = Label(frame_nw_normalized, text="Normalize distance for peak matching:")

nw_normalized_var = StringVar(root)
nw_normalized_var.set(bool_options[1])
nw_normalized_menu = OptionMenu(frame_nw_normalized, nw_normalized_var, *bool_options, command=update_peak_match_window_label)

nw_normalized_label.pack(side=tk.LEFT)
nw_normalized_menu.pack(side=tk.LEFT)

# peak_match_window
frame_peak_match_window = tk.Frame(frame_peak_matchr)
frame_peak_match_window.pack(**padding_setting, side=tk.TOP)

peak_match_window_label = Label(frame_peak_match_window, text="Peak Matching Window:")
peak_match_window_label2 = Label(frame_peak_match_window, text="Hz")

peak_match_window_var = DoubleVar(root)
peak_match_window_entry = Entry(frame_peak_match_window, width=6)
peak_match_window_entry.insert(0, 150)

peak_match_window_label.pack(side=tk.LEFT)
peak_match_window_entry.pack(side=tk.LEFT)
peak_match_window_label2.pack(side=tk.LEFT)

# matching_penalty_order
frame_matching_penalty_order = tk.Frame(frame_peak_matchr)
frame_matching_penalty_order.pack(**padding_setting, side=tk.TOP)

matching_penalty_order_label = Label(frame_matching_penalty_order, text="Matching penalty order:")

matching_penalty_order_var = DoubleVar(root)
matching_penalty_order_entry = Entry(frame_matching_penalty_order, width=6)
matching_penalty_order_entry.insert(0, 1)

matching_penalty_order_label.pack(side=tk.LEFT)
matching_penalty_order_entry.pack(side=tk.LEFT)

# dummy
frame_dummy = tk.Frame(frame_peak_matchr)
frame_dummy.pack(**padding_setting, side=tk.TOP)

dummy_label = Label(frame_dummy, text="")

dummy_label.pack(side=tk.LEFT)

# SAVING

# save
frame_save = tk.Frame(root)
frame_save.pack(**padding_option, side=tk.TOP)
heading("Saving", frame=frame_save)

save_label = Label(frame_save, text="Save data to .pkl files:")
save_label.pack(side=tk.LEFT)

save_var = StringVar(root)
save_var.set(bool_options[1])
save_menu = OptionMenu(frame_save, save_var, *bool_options)
save_menu.pack(side=tk.LEFT)

# save_tag
frame_save_tag = tk.Frame(root)
frame_save_tag.pack(**padding_setting, side=tk.TOP)

save_tag_label = Label(frame_save_tag, text="Save filename: data_dict_... and peak_results")
save_tag_label2 = Label(frame_save_tag, text=".pkl")

save_tag_var = StringVar(root)
save_tag_var.trace_add("write", update_save_tag_label)
save_tag_entry = Entry(frame_save_tag, width=6, textvariable=save_tag_var)
save_tag_entry.insert(0, '')

save_tag_label.pack(side=tk.LEFT)
save_tag_entry.pack(side=tk.LEFT)
save_tag_label2.pack(side=tk.LEFT)


# SUBMIT
frame_submit = tk.Frame(root)
frame_submit.pack(**padding_setting, side=tk.TOP)

submit_button = Button(frame_submit, text="Submit", command=submit)
# submit_button.grid(row=100, column=0, columnspan=100)
submit_button.pack(side=tk.TOP)

# Start the main loop
root.mainloop()


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
"""
