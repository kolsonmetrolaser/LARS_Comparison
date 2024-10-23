# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# Import the tkinter module
import tkinter as tk
from tkinter import filedialog, DoubleVar, StringVar, Label, Entry, Button
from tkinter import OptionMenu, IntVar, Variable, BooleanVar, font
from MetroLaserLARS.LARS_Comparison import LARS_Comparison_from_app


def run_app():
    def make_settings(suppress=False):
        try:
            settings = {}
            # fmt: off
            # DATA
            settings['directory']                  = directory_var.get() # noqa
            settings['pickled_data_path']          = pickled_data_path_var.get() # noqa
            # DATA DEFINITIONS
            settings['frange']                     = (frange_min_var.get()/1000, frange_max_var.get()/1000) # noqa
            settings['slc_limits']                 = (frange_min_var.get(), frange_max_var.get()) # noqa
            settings['plot']                       = directory_var.get() # noqa
            # PLOTTING AND PRINTING
            settings['plot']                       = True if plot_var.get() == 'True' else False  # noqa
            settings['plot_detail']                = True if plot_detail_var.get() == 'True' else False  # noqa
            settings['plot_recursive_noise']       = True if plot_recursive_noise_var.get() == 'True' else False # noqa
            settings['plot_classification']        = True if plot_classification_var.get() == 'True' else False # noqa
            settings['show_plots']                 = True if show_plots_var.get() == 'True' else False  # noqa
            settings['save_plots']                 = True if save_plots_var.get() == 'True' else False  # noqa
            settings['peak_plot_width']            = peak_plot_width_var.get() # noqa
            settings['PRINT_MODE']                 = PRINT_MODE_var.get() # noqa
            # PEAK FITTING
            # baseline removal
            settings['baseline_smoothness']       = 10**baseline_smoothness_var.get() # noqa
            settings['baseline_polyorder']        = baseline_polyorder_var.get() # noqa
            settings['baseline_itermax']          = baseline_itermax_var.get() # noqa
            # smoothing
            settings['sgf_applications']          = sgf_applications_var.get() # noqa
            settings['sgf_windowsize']            = sgf_windowsize_var.get() # noqa
            settings['sgf_polyorder']             = sgf_polyorder_var.get() # noqa
            # peak finding
            settings['peak_height_min']           = peak_height_min_var.get() # noqa
            settings['peak_prominence_min']       = peak_prominence_min_var.get() # noqa
            settings['peak_ph_ratio_min']         = peak_ph_ratio_min_var.get() # noqa
            # noise reduction
            settings['recursive_noise_reduction'] = True if recursive_noise_reduction_var.get() == 'True' else False  # noqa
            settings['max_noise_reduction_iter']  = max_noise_reduction_iter_var.get() # noqa
            settings['regularization_ratio']      = regularization_ratio_var.get() # noqa
            # PEAK MATCHING
            # stretching
            settings['max_stretch']               = max_stretch_var.get() # noqa
            settings['num_stretches']             = num_stretches_var.get() # noqa
            settings['stretching_iterations']     = stretching_iterations_var.get() # noqa
            settings['stretch_iteration_factor']  = stretch_iteration_factor_var.get() # noqa
            # matching
            settings['peak_match_window']         = peak_match_window_var.get() # noqa
            settings['matching_penalty_order']    = matching_penalty_order_var.get() # noqa
            settings['nw_normalized']             = True if nw_normalized_var.get() == 'True' else False  # noqa
            # SAVING
            settings['save_data']                 = True if save_data_var.get() == 'True' else False  # noqa
            settings['save_results']              = True if save_results_var.get() == 'True' else False  # noqa
            settings['save_tag']                  = save_tag_var.get() # noqa
            print(save_tag_var.get())
            settings['save_folder']               = directory_var.get() if save_directory_var.get() == 'Same as LARS Data Directory' else save_directory_var.get() # noqa
            # fmt: on
        except tk.TclError as e:
            if not suppress:
                raise e
            return settings
        except Exception as e:
            raise e

        return settings

    def submit():
        if status_var.get() in ['nodir', 'running']:
            return

        settings = make_settings()
        running_var.set(True)
        update_status()

        LARS_Comparison_from_app(settings)

        prev_settings_var.set(settings)
        running_var.set(False)
        update_status()

    def select_directory():
        directory = filedialog.askdirectory(title="Select a Directory")
        if directory:
            directory_entry.delete(0, tk.END)  # Clear the directory_entry box
            directory_entry.insert(0, directory)  # Insert the selected directory

    def select_save_directory():
        save_directory = filedialog.askdirectory(title="Select a Directory")
        if save_directory:
            save_directory_entry.delete(0, tk.END)  # Clear the save_directory_entry box
            save_directory_entry.insert(0, save_directory)  # Insert the selected save_directory

    def select_pickled_data_path():
        pickled_data_path = filedialog.askopenfilename(title="Select a data_dict[...].pkl file",
                                                       filetypes=[("Pickled Data Dictionaries", "data_dict*.pkl"), ("All Files", "*.*")])
        if pickled_data_path:
            pickled_data_path_entry.delete(0, tk.END)  # Clear the pickled_data_path_entry box
            pickled_data_path_entry.insert(0, pickled_data_path)  # Insert the selected pickled_data_path

    def update_peak_match_window_label(entry_text):
        # Update the label text with the selected option
        txt = "Hz" if entry_text == 'False' else '* peak position / Hz'
        peak_match_window_label2.config(text=txt)

    def update_save_tag_label(*args):
        # Update the label text with the selected option
        txt = "Save filename: data_dict... and peak_results" if save_tag_var.get() == '' else 'Save filename: data_dict_... and peak_results_'
        save_tag_label.config(text=txt)

    def hide_save_tag(*args):
        # update_status()  # also update submit button color
        if save_data_var.get() == 'True' or save_results_var.get() == 'True':
            save_tag_dummy_label.pack_forget()
            save_tag_label.pack(side=tk.LEFT)
            save_tag_entry.pack(side=tk.LEFT)
            save_tag_label2.pack(side=tk.LEFT)
            # save_tag_dummy_label.pack(side=tk.BOTTOM)
        else:
            save_tag_label.pack_forget()
            save_tag_label2.pack_forget()
            save_tag_entry.pack_forget()
            save_tag_dummy_label.pack()

    def update_status(*args):
        if running_var.get():
            status_var.set('running')
        elif Variable(root, make_settings(suppress=True)).get() == prev_settings_var.get():
            status_var.set('ran')
        elif directory_var.get() != '' and save_results_var.get() == 'True':
            status_var.set('ready')
        elif directory_var.get() != '':
            status_var.set('nosave')
        else:
            status_var.set('nodir')

        if status_var.get() == 'ready':
            bg, fg = 'green4', 'white'
            button_text = 'Run Code'
            label_state = tk.NORMAL
            if save_data_var.get() == 'True':
                label_text = 'Ready to run code!'
            else:
                label_text = 'Ready to run code! Results will be saved, but not data.'
        elif status_var.get() == 'nosave':
            bg, fg = 'gold', 'black'
            button_text = 'Run Code'
            label_text = 'Warning: results will not be saved.'
            label_state = tk.NORMAL
        elif status_var.get() == 'nodir':
            bg, fg = 'firebrick4', 'white'
            button_text = 'Run Code'
            label_text = 'No directory selected.'
            label_state = tk.NORMAL
        elif status_var.get() == 'running':
            bg, fg = 'RoyalBlue2', 'white'
            button_text = 'Running Code...'
            label_text = ''
            label_state = tk.DISABLED
        elif status_var.get() == 'ran':
            bg, fg = 'DarkOrange1', 'black'
            button_text = 'Run Code'
            label_text = 'Just ran with these settings.'
            label_state = tk.NORMAL

        submit_button.config(bg=bg, fg=fg, text=button_text)
        status_label.config(bg=bg, fg=fg, text=label_text, state=label_state)

        root.update()
        return

    # Create the main window
    root = tk.Tk()
    root.title("LARS Comparison Settings")

    roottop = tk.Frame(root)
    roottop.pack(side=tk.TOP)
    rootsettings = tk.Frame(root)
    rootsettings.pack(side=tk.TOP)
    rootsubmit = tk.Frame(root)
    rootsubmit.pack(side=tk.BOTTOM)

    rootl = tk.Frame(rootsettings)
    rootl.pack(side=tk.LEFT)
    rootr = tk.Frame(rootsettings)
    rootr.pack(side=tk.LEFT)

    def heading(txt, frame=root, lvl=0, padding=True, side=tk.TOP, subtext=None):
        fonts = [(default_font_name, 20, "bold"), (default_font_name, 14, "bold"), (default_font_name, 10, "bold"), (default_font_name, 9)]
        h = tk.Label(frame, text=txt, font=fonts[lvl])
        if subtext is not None:
            subframe = tk.Frame(frame)
            if padding:
                subframe.pack(**padding_heading, side=side)
            else:
                subframe.pack(side=side)
            h = heading(txt, frame=subframe, lvl=lvl, padding=False, side=tk.TOP)
            h2 = heading(subtext, frame=subframe, lvl=3, padding=False, side=tk.BOTTOM)
            return h, h2
        if padding:
            h.pack(**padding_heading, side=side)
        else:
            h.pack(side=side)
        return h

    # Global info
    bool_options = ['True', 'False']
    padding_heading = {'pady': 10, 'padx': 10}
    padding_setting = {'pady': 4, 'padx': 4}
    padding_option = {'pady': 0, 'padx': 4}
    dashes = '-'*45
    running_var = BooleanVar(root, value=False)
    prev_settings_var = Variable(root, value={})
    status_var = StringVar(root, value='nodir')
    default_font_name = font.nametofont('TkTextFont').actual()['family']

    # Start building App

    heading(dashes+' Load Data '+dashes, frame=roottop, padding=False, side=tk.TOP)
    rootload = tk.Frame(roottop)
    rootload.pack(side=tk.TOP)
    # DIRECTORY

    frame_directory = tk.Frame(rootload)
    frame_directory.pack(**padding_setting, side=tk.TOP)
    heading("Directory", lvl=1, frame=frame_directory,
            subtext="""Select a folder which contains subfolders, each of which contain LARS data in .all format.
    All pairs of subfolders will be compared.""")

    directory_label = Label(frame_directory, text="Enter path to LARS data or select a folder:")

    directory_var = StringVar(root)
    directory_var.trace_add("write", update_status)
    directory_entry = Entry(frame_directory, width=40, textvariable=directory_var)

    directory_button = Button(frame_directory, text="Open", command=select_directory, bg='gray75')

    directory_label.pack(side=tk.LEFT, padx=4)
    directory_button.pack(side=tk.LEFT, padx=4)
    directory_entry.pack(side=tk.LEFT, padx=4)

    # frame_or = tk.Frame(rootload)
    # frame_or.pack(**padding_setting, side=tk.LEFT)
    # heading("OR", lvl=1, frame=frame_or, padding=padding_heading, side=tk.TOP)
    # or_label = Label(frame_or, text='')
    # or_label.pack(side=tk.BOTTOM)

    # pickled_data_path

    frame_pickled_data_path = tk.Frame(rootload)
    frame_pickled_data_path.pack(**padding_setting, side=tk.BOTTOM)
    heading("Pickled data", lvl=1, frame=frame_pickled_data_path, subtext='Load data from previous analysis.')

    pickled_data_path_label2 = Label(frame_pickled_data_path, text="Enter path to pickled data or select a folder:")

    pickled_data_path_var = StringVar(root)
    pickled_data_path_var.trace_add("write", update_status)
    pickled_data_path_entry = Entry(frame_pickled_data_path, width=40, textvariable=pickled_data_path_var)

    pickled_data_path_button = Button(frame_pickled_data_path, text="Open", command=select_pickled_data_path, bg='gray75')

    pickled_data_path_label2.pack(side=tk.LEFT, padx=4)
    pickled_data_path_button.pack(side=tk.LEFT, padx=4)
    pickled_data_path_entry.pack(side=tk.LEFT, padx=4)

    heading('-'+dashes+' Settings '+dashes+'-', frame=roottop, side=tk.BOTTOM)

    # DATA DEFINITIONS

    # frange and slc limits
    frame_frange = tk.Frame(rootl)
    frame_frange.pack(**padding_setting, side=tk.TOP)
    heading("Data", lvl=1, frame=frame_frange)

    frange_label = Label(frame_frange, text="The data minimum and maximum frequency:")
    frange_label2 = Label(frame_frange, text="-")
    frange_label3 = Label(frame_frange, text="Hz")

    frange_min_var, frange_max_var = DoubleVar(root, value=10000), DoubleVar(root, value=60000)
    frange_min_var.trace_add("write", update_status)
    frange_max_var.trace_add("write", update_status)
    frange_min_entry = Entry(frame_frange, width=6, textvariable=frange_min_var)
    frange_max_entry = Entry(frame_frange, width=6, textvariable=frange_max_var)

    frange_label.pack(side=tk.LEFT)
    frange_min_entry.pack(side=tk.LEFT)
    frange_label2.pack(side=tk.LEFT)
    frange_max_entry.pack(side=tk.LEFT)
    frange_label3.pack(side=tk.LEFT)

    # combine
    frame_combine = tk.Frame(rootl)
    frame_combine.pack(**padding_option, side=tk.TOP)

    combine_label = Label(frame_combine, text="How data within a folder should be combined:")

    combine_options = ['max', 'mean']
    combine_var = StringVar(root, value=combine_options[0])
    combine_var.trace_add("write", update_status)
    combine_menu = OptionMenu(frame_combine, combine_var, *combine_options)
    combine_menu.config(bg='gray75')

    combine_label.pack(side=tk.LEFT)
    combine_menu.pack(side=tk.LEFT)

    # PLOTTING AND PRINTING

    # plot, plot_detail, plot_recursive_noise and plot_classification
    frame_plot = tk.Frame(rootl)
    frame_plot.pack(**padding_option, side=tk.TOP)

    heading("Plotting and Printing", lvl=1, frame=frame_plot, side=tk.TOP)

    frame_plot_label = tk.Frame(frame_plot)
    frame_plot_label.pack(side=tk.TOP)

    plot_label = Label(frame_plot_label, text="Create plots (with fitting details)\n(of recursive noise iterations) (of classification):")
    plot_label.pack(side=tk.BOTTOM)

    frame_plot_menus = tk.Frame(frame_plot)
    frame_plot_menus.pack(side=tk.BOTTOM)

    plot_var = StringVar(root, value=bool_options[1])
    plot_var.trace_add("write", update_status)
    plot_menu = OptionMenu(frame_plot_menus, plot_var, *bool_options)
    plot_menu.config(bg='gray75')
    plot_menu.grid(row=0, column=0)
    plot_detail_var = StringVar(root, value=bool_options[1])
    plot_detail_var.trace_add("write", update_status)
    plot_detail_menu = OptionMenu(frame_plot_menus, plot_detail_var, *bool_options)
    plot_detail_menu.config(bg='gray75')
    plot_detail_menu.grid(row=0, column=1)
    plot_recursive_noise_var = StringVar(root, value=bool_options[1])
    plot_recursive_noise_var.trace_add("write", update_status)
    plot_recursive_noise_menu = OptionMenu(frame_plot_menus, plot_recursive_noise_var, *bool_options)
    plot_recursive_noise_menu.config(bg='gray75')
    plot_recursive_noise_menu.grid(row=1, column=0)
    plot_classification_var = StringVar(root, value=bool_options[1])
    plot_classification_var.trace_add("write", update_status)
    plot_classification_menu = OptionMenu(frame_plot_menus, plot_classification_var, *bool_options)
    plot_classification_menu.config(bg='gray75')
    plot_classification_menu.grid(row=1, column=1)

    # peak_plot_width
    frame_peak_fit_plot_width = tk.Frame(rootl)
    frame_peak_fit_plot_width.pack(**padding_setting, side=tk.TOP)

    peak_plot_width_label = Label(frame_peak_fit_plot_width, text="Width of peak fit plots:")
    peak_plot_width_label2 = Label(frame_peak_fit_plot_width, text="kHz")

    peak_plot_width_var = DoubleVar(root, value=20)
    peak_plot_width_var.trace_add("write", update_status)
    peak_plot_width_entry = Entry(frame_peak_fit_plot_width, width=6, textvariable=peak_plot_width_var)

    peak_plot_width_label.pack(side=tk.LEFT)
    peak_plot_width_entry.pack(side=tk.LEFT)
    peak_plot_width_label2.pack(side=tk.LEFT)

    # show_plots
    frame_show_save_plots = tk.Frame(rootl)
    frame_show_save_plots.pack(**padding_option, side=tk.TOP)

    show_plots_label = Label(frame_show_save_plots, text="Show plots:")

    show_plots_var = StringVar(root, value=bool_options[1])
    show_plots_var.trace_add("write", update_status)
    show_plots_menu = OptionMenu(frame_show_save_plots, show_plots_var, *bool_options, command=hide_save_tag)
    show_plots_menu.config(bg='gray75')

    show_plots_label.pack(side=tk.LEFT)
    show_plots_menu.pack(**padding_setting, side=tk.LEFT)

    # save_plots
    save_plots_label = Label(frame_show_save_plots, text="Save plots:")

    save_plots_var = StringVar(root, value=bool_options[1])
    save_plots_var.trace_add("write", update_status)
    save_plots_menu = OptionMenu(frame_show_save_plots, save_plots_var, *bool_options, command=hide_save_tag)
    save_plots_menu.config(bg='gray75')

    save_plots_label.pack(**padding_setting, side=tk.LEFT)
    save_plots_menu.pack(side=tk.LEFT)

    # PRINT_MODE
    frame_PRINT_MODE = tk.Frame(rootl)
    frame_PRINT_MODE.pack(**padding_option, side=tk.TOP)

    PRINT_MODE_options = ['none', 'sparse', 'full']
    PRINT_MODE_label = Label(frame_PRINT_MODE, text="Print details:")

    PRINT_MODE_var = StringVar(root, value=PRINT_MODE_options[1])
    PRINT_MODE_var.trace_add("write", update_status)
    PRINT_MODE_menu = OptionMenu(frame_PRINT_MODE, PRINT_MODE_var, *PRINT_MODE_options)
    PRINT_MODE_menu.config(bg='gray75')

    PRINT_MODE_label.pack(side=tk.LEFT)
    PRINT_MODE_menu.pack(side=tk.LEFT)

    # SAVING

    # save_data
    frame_save_data = tk.Frame(rootl)
    frame_save_data.pack(**padding_option, side=tk.TOP)
    heading("Saving", lvl=1, frame=frame_save_data)

    save_data_label = Label(frame_save_data, text="Save data to .pkl file:")
    save_data_label.pack(side=tk.LEFT)

    save_data_var = StringVar(root, value=bool_options[1])
    save_data_var.trace_add("write", update_status)
    save_data_menu = OptionMenu(frame_save_data, save_data_var, *bool_options, command=hide_save_tag)
    save_data_menu.config(bg='gray75')
    save_data_menu.pack(side=tk.LEFT)

    # save_results
    frame_save_results = tk.Frame(rootl)
    frame_save_results.pack(**padding_option, side=tk.TOP)

    save_results_label = Label(frame_save_results, text="Save results to .pkl file:")
    save_results_label.pack(side=tk.LEFT)

    save_results_var = StringVar(root, value=bool_options[0])
    save_results_var.trace_add("write", update_status)
    save_results_menu = OptionMenu(frame_save_results, save_results_var, *bool_options, command=hide_save_tag)
    save_results_menu.config(bg='gray75')
    save_results_menu.pack(side=tk.LEFT)

    # save_tag
    frame_save_tag = tk.Frame(rootl)
    frame_save_tag.pack(**padding_setting, side=tk.TOP)

    save_tag_label = Label(frame_save_tag, text="Save filename: data_dict... and peak_results")
    save_tag_label2 = Label(frame_save_tag, text=".pkl")

    save_tag_var = StringVar(root, value='')
    save_tag_var.trace_add("write", update_save_tag_label)
    save_tag_var.trace_add("write", update_status)
    save_tag_entry = Entry(frame_save_tag, width=6, textvariable=save_tag_var)

    save_tag_dummy_label = Label(frame_save_tag, text=" "*42, font=("Courier New", 9))
    # save_tag_dummy_label.pack()

    save_tag_label.pack(side=tk.LEFT)
    save_tag_entry.pack(side=tk.LEFT)
    save_tag_label2.pack(side=tk.LEFT)

    # save_directory

    frame_save_directory_label = tk.Frame(rootl)
    frame_save_directory_label.pack(**padding_setting, side=tk.TOP)
    frame_save_directory = tk.Frame(rootl)
    frame_save_directory.pack(**padding_setting, side=tk.TOP)

    save_directory_label = Label(frame_save_directory_label, text="Enter path to save data to or select a folder:")

    save_directory_var = StringVar(root, value='Same as LARS Data Directory')
    save_directory_var.trace_add("write", update_status)
    save_directory_entry = Entry(frame_save_directory, width=40, textvariable=save_directory_var)

    save_directory_button = Button(frame_save_directory, text="Open", command=select_save_directory, bg='gray75')

    save_directory_label.pack(side=tk.LEFT, padx=4)
    save_directory_button.pack(side=tk.LEFT, padx=4)
    save_directory_entry.pack(side=tk.LEFT, padx=4)

    # PEAK FITTING
    frame_peak_fit = tk.Frame(rootr)
    frame_peak_fit.pack(side=tk.TOP)
    heading("Peak Fitting", lvl=1, frame=frame_peak_fit, padding=False)

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

    baseline_smoothness_var = DoubleVar(root, value=12)
    baseline_smoothness_var.trace_add("write", update_status)
    baseline_smoothness_entry = Entry(frame_baseline_smoothness, width=6, textvariable=baseline_smoothness_var)

    baseline_smoothness_label.pack(side=tk.LEFT)
    baseline_smoothness_entry.pack(side=tk.LEFT)

    # baseline_polyorder
    frame_baseline_polyorder = tk.Frame(frame_peak_fitl)
    frame_baseline_polyorder.pack(**padding_setting, side=tk.TOP)

    baseline_polyorder_label = Label(frame_baseline_polyorder, text="Basline polyorder:")

    baseline_polyorder_var = IntVar(root, value=2)
    baseline_polyorder_var.trace_add("write", update_status)
    baseline_polyorder_entry = Entry(frame_baseline_polyorder, width=6, textvariable=baseline_polyorder_var)

    baseline_polyorder_label.pack(side=tk.LEFT)
    baseline_polyorder_entry.pack(side=tk.LEFT)

    # baseline_itermax
    frame_baseline_itermax = tk.Frame(frame_peak_fitl)
    frame_baseline_itermax.pack(**padding_setting, side=tk.TOP)

    baseline_itermax_label = Label(frame_baseline_itermax, text="Basline itermax:")

    baseline_itermax_var = IntVar(root, value=10)
    baseline_itermax_var.trace_add("write", update_status)
    baseline_itermax_entry = Entry(frame_baseline_itermax, width=6, textvariable=baseline_itermax_var)

    baseline_itermax_label.pack(side=tk.LEFT)
    baseline_itermax_entry.pack(side=tk.LEFT)

    # sgf_windowsize
    frame_sgf_windowsize = tk.Frame(frame_peak_fitr)
    frame_sgf_windowsize.pack(**padding_setting, side=tk.TOP)

    sgf_windowsize_label = Label(frame_sgf_windowsize, text="SGF Window Size:")

    sgf_windowsize_var = IntVar(root, value=101)
    sgf_windowsize_var.trace_add("write", update_status)
    sgf_windowsize_entry = Entry(frame_sgf_windowsize, width=6, textvariable=sgf_windowsize_var)

    sgf_windowsize_label.pack(side=tk.LEFT)
    sgf_windowsize_entry.pack(side=tk.LEFT)

    # sgf_applications
    frame_sgf_applications = tk.Frame(frame_peak_fitr)
    frame_sgf_applications.pack(**padding_setting, side=tk.TOP)

    sgf_applications_label = Label(frame_sgf_applications, text="SGF Applications:")

    sgf_applications_var = IntVar(root, value=2)
    sgf_applications_var.trace_add("write", update_status)
    sgf_applications_entry = Entry(frame_sgf_applications, width=6, textvariable=sgf_applications_var)

    sgf_applications_label.pack(side=tk.LEFT)
    sgf_applications_entry.pack(side=tk.LEFT)

    # sgf_polyorder
    frame_sgf_polyorder = tk.Frame(frame_peak_fitr)
    frame_sgf_polyorder.pack(**padding_setting, side=tk.TOP)

    sgf_polyorder_label = Label(frame_sgf_polyorder, text="SGF polyorder:")

    sgf_polyorder_var = IntVar(root, value=0)
    sgf_polyorder_var.trace_add("write", update_status)
    sgf_polyorder_entry = Entry(frame_sgf_polyorder, width=6, textvariable=sgf_polyorder_var)

    sgf_polyorder_label.pack(side=tk.LEFT)
    sgf_polyorder_entry.pack(side=tk.LEFT)

    # headings
    heading("Peak Finding", lvl=2, frame=frame_peak_fitl, padding=False)
    heading("Noise Reduction", lvl=2, frame=frame_peak_fitr, padding=False)

    # peak_height_min
    frame_peak_fit_height_min = tk.Frame(frame_peak_fitl)
    frame_peak_fit_height_min.pack(**padding_setting, side=tk.TOP)

    peak_height_min_label = Label(frame_peak_fit_height_min, text="Peak height minimum: noise *")

    peak_height_min_var = DoubleVar(root, value=0.2)
    peak_height_min_var.trace_add("write", update_status)
    peak_height_min_entry = Entry(frame_peak_fit_height_min, width=6, textvariable=peak_height_min_var)

    peak_height_min_label.pack(side=tk.LEFT)
    peak_height_min_entry.pack(side=tk.LEFT)

    # peak_prominence_min
    frame_peak_fit_prominence_min = tk.Frame(frame_peak_fitl)
    frame_peak_fit_prominence_min.pack(**padding_setting, side=tk.TOP)

    peak_prominence_min_label = Label(frame_peak_fit_prominence_min, text="Peak prominence minimum: noise *")

    peak_prominence_min_var = DoubleVar(root, value=0.2)
    peak_prominence_min_var.trace_add("write", update_status)
    peak_prominence_min_entry = Entry(frame_peak_fit_prominence_min, width=6, textvariable=peak_prominence_min_var)

    peak_prominence_min_label.pack(side=tk.LEFT)
    peak_prominence_min_entry.pack(side=tk.LEFT)

    # peak_ph_ratio_min
    frame_peak_fit_ph_ratio_min = tk.Frame(frame_peak_fitl)
    frame_peak_fit_ph_ratio_min.pack(**padding_setting, side=tk.TOP)

    peak_ph_ratio_min_label = Label(frame_peak_fit_ph_ratio_min, text="Peak prominence-to-height minimum:")

    peak_ph_ratio_min_var = DoubleVar(root, value=0.5)
    peak_ph_ratio_min_var.trace_add("write", update_status)
    peak_ph_ratio_min_entry = Entry(frame_peak_fit_ph_ratio_min, width=6, textvariable=peak_ph_ratio_min_var)

    peak_ph_ratio_min_label.pack(side=tk.LEFT)
    peak_ph_ratio_min_entry.pack(side=tk.LEFT)

    # recursive_noise_reduction
    frame_recursive_noise_reduction = tk.Frame(frame_peak_fitr)
    frame_recursive_noise_reduction.pack(**padding_option, side=tk.TOP)

    recursive_noise_reduction_label = Label(frame_recursive_noise_reduction, text="Recursively reduce noise:")

    recursive_noise_reduction_var = StringVar(root, value=bool_options[0])
    recursive_noise_reduction_var.trace_add("write", update_status)
    recursive_noise_reduction_menu = OptionMenu(frame_recursive_noise_reduction, recursive_noise_reduction_var,
                                                *bool_options)
    recursive_noise_reduction_menu.config(bg='gray75')

    recursive_noise_reduction_label.pack(side=tk.LEFT)
    recursive_noise_reduction_menu.pack(side=tk.LEFT)

    # max_noise_reduction_iter
    frame_max_noise_reduction_iter = tk.Frame(frame_peak_fitr)
    frame_max_noise_reduction_iter.pack(**padding_setting, side=tk.TOP)

    max_noise_reduction_iter_label = Label(frame_max_noise_reduction_iter, text="Max noise reduction iterations:")

    max_noise_reduction_iter_var = IntVar(root, value=10)
    max_noise_reduction_iter_var.trace_add("write", update_status)
    max_noise_reduction_iter_entry = Entry(frame_max_noise_reduction_iter, width=6,
                                           textvariable=max_noise_reduction_iter_var)

    max_noise_reduction_iter_label.pack(side=tk.LEFT)
    max_noise_reduction_iter_entry.pack(side=tk.LEFT)

    # regularization_ratio
    frame_regularization_ratio = tk.Frame(frame_peak_fitr)
    frame_regularization_ratio.pack(**padding_setting, side=tk.TOP)

    regularization_ratio_label = Label(frame_regularization_ratio, text="Noise reduction regularization factor:")

    regularization_ratio_var = DoubleVar(root, value=0.5)
    regularization_ratio_var.trace_add("write", update_status)
    regularization_ratio_entry = Entry(frame_regularization_ratio, width=6, textvariable=regularization_ratio_var)

    regularization_ratio_label.pack(side=tk.LEFT)
    regularization_ratio_entry.pack(side=tk.LEFT)

    # PEAK MATCHING
    frame_peak_match = tk.Frame(rootr)
    frame_peak_match.pack(side=tk.TOP)
    heading("Peak Matching", lvl=1, frame=frame_peak_match)

    frame_peak_matchl = tk.Frame(frame_peak_match)
    frame_peak_matchl.pack(side=tk.LEFT)
    frame_peak_matchr = tk.Frame(frame_peak_match)
    frame_peak_matchr.pack(side=tk.LEFT)
    heading("Stretching", lvl=2, frame=frame_peak_matchl, padding=False)
    heading("Matching", lvl=2, frame=frame_peak_matchr, padding=False)

    # max_stretch
    frame_max_stretch = tk.Frame(frame_peak_matchl)
    frame_max_stretch.pack(**padding_setting, side=tk.TOP)

    max_stretch_label = Label(frame_max_stretch, text="Max stretching: 1 ±")

    max_stretch_var = DoubleVar(root, value=0.02)
    max_stretch_var.trace_add("write", update_status)
    max_stretch_entry = Entry(frame_max_stretch, width=6, textvariable=max_stretch_var)

    max_stretch_label.pack(side=tk.LEFT)
    max_stretch_entry.pack(side=tk.LEFT)

    # num_stretches
    frame_num_stretches = tk.Frame(frame_peak_matchl)
    frame_num_stretches.pack(**padding_setting, side=tk.TOP)

    num_stretches_label = Label(frame_num_stretches, text="Number of stretches per iteration:")

    num_stretches_var = IntVar(root, value=1000)
    num_stretches_var.trace_add("write", update_status)
    num_stretches_entry = Entry(frame_num_stretches, width=6, textvariable=num_stretches_var)

    num_stretches_label.pack(side=tk.LEFT)
    num_stretches_entry.pack(side=tk.LEFT)

    # stretching_iterations
    frame_stretching_iterations = tk.Frame(frame_peak_matchl)
    frame_stretching_iterations.pack(**padding_setting, side=tk.TOP)

    stretching_iterations_label = Label(frame_stretching_iterations, text="Number of stretching iterations:")

    stretching_iterations_var = IntVar(root, value=10)
    stretching_iterations_var.trace_add("write", update_status)
    stretching_iterations_entry = Entry(frame_stretching_iterations, width=6, textvariable=stretching_iterations_var)

    stretching_iterations_label.pack(side=tk.LEFT)
    stretching_iterations_entry.pack(side=tk.LEFT)

    # stretch_iteration_factor
    frame_stretch_iteration_factor = tk.Frame(frame_peak_matchl)
    frame_stretch_iteration_factor.pack(**padding_setting, side=tk.TOP)

    stretch_iteration_factor_label = Label(frame_stretch_iteration_factor,
                                           text="Factor to reduce stretch space each iteration:")

    stretch_iteration_factor_var = DoubleVar(root, value=5)
    stretch_iteration_factor_var.trace_add("write", update_status)
    stretch_iteration_factor_entry = Entry(frame_stretch_iteration_factor, width=6,
                                           textvariable=stretch_iteration_factor_var)

    stretch_iteration_factor_label.pack(side=tk.LEFT)
    stretch_iteration_factor_entry.pack(side=tk.LEFT)

    # nw_normalized
    frame_nw_normalized = tk.Frame(frame_peak_matchr)
    frame_nw_normalized.pack(**padding_option, side=tk.TOP)

    nw_normalized_label = Label(frame_nw_normalized, text="Normalize distance for peak matching:")

    nw_normalized_var = StringVar(root)
    nw_normalized_var.set(bool_options[1])
    nw_normalized_menu = OptionMenu(frame_nw_normalized, nw_normalized_var, *bool_options,
                                    command=update_peak_match_window_label)
    nw_normalized_menu.config(bg='gray75')

    nw_normalized_label.pack(side=tk.LEFT)
    nw_normalized_menu.pack(side=tk.LEFT)

    # peak_match_window
    frame_peak_match_window = tk.Frame(frame_peak_matchr)
    frame_peak_match_window.pack(**padding_setting, side=tk.TOP)

    peak_match_window_label = Label(frame_peak_match_window, text="Peak Matching Window:")
    peak_match_window_label2 = Label(frame_peak_match_window, text="Hz")

    peak_match_window_var = DoubleVar(root, value=150)
    peak_match_window_var.trace_add("write", update_status)
    peak_match_window_entry = Entry(frame_peak_match_window, width=6, textvariable=peak_match_window_var)

    peak_match_window_label.pack(side=tk.LEFT)
    peak_match_window_entry.pack(side=tk.LEFT)
    peak_match_window_label2.pack(side=tk.LEFT)

    # matching_penalty_order
    frame_matching_penalty_order = tk.Frame(frame_peak_matchr)
    frame_matching_penalty_order.pack(**padding_setting, side=tk.TOP)

    matching_penalty_order_label = Label(frame_matching_penalty_order, text="Matching penalty order:")

    matching_penalty_order_var = DoubleVar(root, value=1)
    matching_penalty_order_var.trace_add("write", update_status)
    matching_penalty_order_entry = Entry(frame_matching_penalty_order, width=6, textvariable=matching_penalty_order_var)

    matching_penalty_order_label.pack(side=tk.LEFT)
    matching_penalty_order_entry.pack(side=tk.LEFT)

    # dummy
    frame_dummy = tk.Frame(frame_peak_matchr)
    frame_dummy.pack(**padding_setting, side=tk.TOP)

    dummy_label = Label(frame_dummy, text="")

    dummy_label.pack(side=tk.LEFT)

    # SUBMIT
    frame_submit = tk.Frame(rootsubmit)
    frame_submit.pack(**padding_heading, side=tk.BOTTOM)

    submit_button = Button(root, text="Run Code", bg='firebrick4', fg='white', width=20, height=2, command=submit, font=(default_font_name, 20, "bold"))
    submit_button.pack(**padding_heading)

    frame_status = tk.Frame(rootsubmit)
    frame_status.pack(side=tk.BOTTOM)

    status_label0 = Label(frame_status, text='Status:', font=(default_font_name, 12))
    status_label0.pack(**padding_setting, side=tk.LEFT)
    status_label = Label(frame_status, text='No directory selected.', bg='firebrick4', fg='white', font=(default_font_name, 12))
    status_label.pack(**padding_setting, side=tk.RIGHT)

    # Start the main loop
    root.mainloop()


if __name__ == '__main__':
    from LARS_Comparison import LARS_Comparison_from_app
    run_app()
