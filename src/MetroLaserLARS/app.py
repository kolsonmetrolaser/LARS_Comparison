# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# External imports
import tkinter as tk
import pickle
from numpy import log10

# Internal imports
try:
    from infotext import infotext
    from LARS_Comparison import LARS_Comparison_from_app
    from app_helpers import heading, labeled_options, labeled_file_select, labeled_entry
    from app_helpers import padding_none, padding_setting, padding_option, padding_heading
    from app_helpers import bool_options, CustomVar
    from app_window_part_matching import open_part_matching_window
    from app_window_plot import open_plot_window
except ModuleNotFoundError:
    from MetroLaserLARS.infotext import infotext
    from MetroLaserLARS.LARS_Comparison import LARS_Comparison_from_app
    from MetroLaserLARS.app_helpers import heading, labeled_options, labeled_file_select, labeled_entry
    from MetroLaserLARS.app_helpers import padding_none, padding_setting, padding_option, padding_heading
    from MetroLaserLARS.app_helpers import bool_options, CustomVar
    from MetroLaserLARS.app_window_part_matching import open_part_matching_window
    from MetroLaserLARS.app_window_plot import open_plot_window


def run_app():
    def make_settings(suppress=False):
        try:
            settings = {}
            # fmt: off
            # DATA
            settings['directory']                 = directory_var.get() # noqa
            settings['data_format']               = data_format_var.get() # noqa
            settings['new_data_format']           = new_data_format_var.get() # noqa
            settings['pickled_data_path']         = pickled_data_path_var.get() # noqa
            # DATA DEFINITIONS
            settings['frange']                    = (frange_min_var.get()/1000, frange_max_var.get()/1000) # noqa
            settings['slc_limits']                = (frange_min_var.get(), frange_max_var.get()) # noqa
            settings['combine']                   = combine_var.get() # noqa
            settings['grouped_folders']           = True if grouped_folders_var.get() == 'True' else False # noqa
            settings['part_matching_text']        = part_matching_text_var.get() # noqa
            settings['part_matching_strategy']    = part_matching_strategy_var.get() # noqa
            # PLOTTING AND PRINTING
            settings['plot']                      = True if plot_var.get() == 'True' else False  # noqa
            settings['plot_detail']               = True if plot_detail_var.get() == 'True' else False  # noqa
            settings['plot_recursive_noise']      = True if plot_recursive_noise_var.get() == 'True' else False # noqa
            settings['plot_classification']       = True if plot_classification_var.get() == 'True' else False # noqa
            settings['show_plots']                = True if show_plots_var.get() == 'True' else False  # noqa
            settings['save_plots']                = True if save_plots_var.get() == 'True' else False  # noqa
            settings['peak_plot_width']           = peak_plot_width_var.get() # noqa
            settings['PRINT_MODE']                = PRINT_MODE_var.get() # noqa
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
            settings['save_folder']               = directory_var.get() if save_directory_var.get() == 'Same as LARS Data Directory' else save_directory_var.get() # noqa
            settings['save_settings']             = settings['save_data'] # True if save_settings_var.get() == 'True' else False # noqa
            # APP INTERACTION
            settings['status_label']              = status_label # noqa
            # fmt: on
        except tk.TclError as e:
            if not suppress:
                raise e
            return settings
        except Exception as e:
            raise e

        return settings

    def set_settings(settings, suppress=False):
        try:
            # fmt: off
            # DATA
            directory_var.set(                 settings['directory'] if 'directory' in settings else '') # noqa
            data_format_var.set(               settings['data_format'] if 'data_format' in settings else 'auto') # noqa
            new_data_format_var.set(           settings['new_data_format'] if 'new_data_format' in settings else 'none') # noqa
            pickled_data_path_var.set(         settings['pickled_data_path'] if 'pickled_data_path' in settings else '') # noqa
            # DATA DEFINITIONS
            frange_min_var.set(                settings['slc_limits'][0] if 'slc_limits' in settings else 10000) # noqa
            frange_max_var.set(                settings['slc_limits'][1] if 'slc_limits' in settings else 60000) # noqa
            combine_var.set(                   settings['combine'] if 'combine' in settings else 'max') # noqa
            grouped_folders_var.set(           ('True' if settings['grouped_folders'] else 'False') if 'grouped_folders' in settings else 'False') # noqa
            part_matching_text_var.set(        settings['part_matching_text'] if 'part_matching_text' in settings else '') # noqa
            part_matching_strategy_var.set(    settings['part_matching_strategy'] if 'part_matching_strategy' in settings else 'list') # noqa
            # PLOTTING AND PRINTING
            plot_var.set(                      ('True' if settings['plot'] else 'False') if 'plot' in settings else 'False') # noqa
            plot_detail_var.set(               ('True' if settings['plot_detail'] else 'False') if 'plot_detail' in settings else 'False') # noqa
            plot_recursive_noise_var.set(      ('True' if settings['plot_recursive_noise'] else 'False') if 'plot_recursive_noise' in settings else 'False') # noqa
            plot_classification_var.set(       ('True' if settings['plot_classification'] else 'False') if 'plot_classification' in settings else 'False') # noqa
            show_plots_var.set(                ('True' if settings['show_plots'] else 'False') if 'show_plots' in settings else 'False') # noqa
            save_plots_var.set(                ('True' if settings['save_plots'] else 'False') if 'save_plots' in settings else 'False') # noqa
            peak_plot_width_var.set(           settings['peak_plot_width'] if 'peak_plot_width' in settings else 20) # noqa
            PRINT_MODE_var.set(                settings['PRINT_MODE'] if 'PRINT_MODE' in settings else 'sparse') # noqa
            # PEAK FITTING
            # baseline removal
            baseline_smoothness_var.set(       log10(settings['baseline_smoothness']) if 'baseline_smoothness' in settings else 12) # noqa
            baseline_polyorder_var.set(        settings['baseline_polyorder'] if 'baseline_polyorder' in settings else 2) # noqa
            baseline_itermax_var.set(          settings['baseline_itermax'] if 'baseline_itermax' in settings else 10) # noqa
            # smoothing
            sgf_applications_var.set(          settings['sgf_applications'] if 'sgf_applications' in settings else 2) # noqa
            sgf_windowsize_var.set(            settings['sgf_windowsize'] if 'sgf_windowsize' in settings else 101) # noqa
            sgf_polyorder_var.set(             settings['sgf_polyorder'] if 'sgf_polyorder' in settings else 0) # noqa
            # peak finding
            peak_height_min_var.set(           settings['peak_height_min'] if 'peak_height_min' in settings else 0.2) # noqa
            peak_prominence_min_var.set(       settings['peak_prominence_min'] if 'peak_prominence_min' in settings else 0.2) # noqa
            peak_ph_ratio_min_var.set(         settings['peak_ph_ratio_min'] if 'peak_ph_ratio_min' in settings else 0.5) # noqa
            # noise reduction
            recursive_noise_reduction_var.set( ('True' if settings['recursive_noise_reduction'] else 'False') if 'recursive_noise_reduction' in settings else 'True') # noqa
            max_noise_reduction_iter_var.set(  settings['max_noise_reduction_iter'] if 'max_noise_reduction_iter' in settings else 10) # noqa
            regularization_ratio_var.set(      settings['regularization_ratio'] if 'regularization_ratio' in settings else 0.5) # noqa
            # PEAK MATCHING
            # stretching
            max_stretch_var.set(               settings['max_stretch'] if 'max_stretch' in settings else 0.02) # noqa
            num_stretches_var.set(             settings['num_stretches'] if 'num_stretches' in settings else 1000) # noqa
            stretching_iterations_var.set(     settings['stretching_iterations'] if 'stretching_iterations' in settings else 10) # noqa
            stretch_iteration_factor_var.set(  settings['stretch_iteration_factor'] if 'stretch_iteration_factor' in settings else 5) # noqa
            # matching
            peak_match_window_var.set(         settings['peak_match_window'] if 'peak_match_window' in settings else 150) # noqa
            matching_penalty_order_var.set(    settings['matching_penalty_order'] if 'matching_penalty_order' in settings else 1) # noqa
            nw_normalized_var.set(             ('True' if settings['nw_normalized'] else 'False') if 'nw_normalized' in settings else 'False') # noqa
            # SAVING
            save_data_var.set(                 ('True' if settings['save_data'] else 'False') if 'save_data' in settings else 'False') # noqa
            save_results_var.set(              ('True' if settings['save_results'] else 'False') if 'save_results' in settings else 'True') # noqa
            save_tag_var.set(                  settings['save_tag'] if 'save_tag' in settings else '') # noqa
            save_directory_var.set(            'Same as LARS Data Directory' if ('save_folder' in settings and 'directory' in settings and settings['save_folder'] == settings['directory']) else (settings['save_folder'] if 'save_folder' in settings else 'Same as LARS Data Directory')) # noqa
            # fmt: on
        except tk.TclError as e:
            if not suppress:
                raise e
            return settings
        except Exception as e:
            raise e

        return settings

    def update_status(*args):
        if running_var.get():
            status_var.set('running')
        elif tk.Variable(root, make_settings(suppress=True)).get() == prev_settings_var.get():
            status_var.set('ran')
        elif directory_var.get() != '' and save_results_var.get() == 'True':
            status_var.set('ready')
        elif directory_var.get() != '':
            status_var.set('nosave')
        else:
            status_var.set('nodir')

        if status_var.get() == 'ready':
            bg, fg = 'green4', 'white'
            tk.Button_text = 'Run Code'
            label_state = tk.NORMAL
            if save_data_var.get() == 'True':
                label_text = 'Ready to run code!'
            else:
                label_text = 'Ready to run code! Results will be saved, but not data.'
        elif status_var.get() == 'nosave':
            bg, fg = 'gold', 'black'
            tk.Button_text = 'Run Code'
            label_text = 'Warning: results will not be saved.'
            label_state = tk.NORMAL
        elif status_var.get() == 'nodir':
            bg, fg = 'firebrick4', 'white'
            tk.Button_text = 'Run Code'
            label_text = 'No directory selected.'
            label_state = tk.NORMAL
        elif status_var.get() == 'running':
            bg, fg = 'RoyalBlue2', 'white'
            tk.Button_text = 'Running Code...'
            label_text = ''
            label_state = tk.DISABLED
        elif status_var.get() == 'ran':
            bg, fg = 'DarkOrange1', 'black'
            tk.Button_text = 'Run Code'
            label_text = 'Just ran with these settings.'
            label_state = tk.NORMAL

        submit_Button.config(bg=bg, fg=fg, text=tk.Button_text)
        status_label.config(bg=bg, fg=fg, text=label_text, state=label_state)

        root.update()
        return

    def submit():
        if status_var.get() in ['nodir', 'running']:
            return

        settings = make_settings()
        running_var.set(True)
        update_status()

        data_dict, pair_results = LARS_Comparison_from_app(settings)
        data_dict_var.set(data_dict)
        pair_results_var.set(pair_results)

        prev_settings_var.set(settings)
        running_var.set(False)
        update_status()

    def import_settings(*args, **kwargs):
        path = tk.filedialog.askopenfilename(**kwargs, title='Choose settings to import',
                                             filetypes=(('Settings Files', '*settings*.pkl'), ('All Files', '*')))
        if path:
            with open(path, 'rb') as f:
                settings_import = pickle.load(f)
                set_settings(settings_import)
        return

    def update_peak_match_window_label(entry_text):
        # Update the label text with the selected option
        peak_match_window_infobox.pack_forget()
        txt = "Hz" if entry_text == 'False' else '*peak_position/Hz'
        peak_match_window_label2.config(text=txt)
        peak_match_window_infobox.pack(side=tk.LEFT)

    def update_save_tag_label(*args):
        # Update the label text with the selected option
        txt = "Save filename: data_dict... and peak_results" if save_tag_var.get(
        ) == '' else 'Save filename: data_dict_... and peak_results_'
        save_tag_label.config(text=txt)

    def hide_save_tag(*args):
        # update_status()  # also update submit tk.Button color
        if save_data_var.get() == 'True' or save_results_var.get() == 'True':
            save_tag_dummy_label.pack_forget()

            save_tag_label.pack(side=tk.LEFT)
            save_tag_entry.pack(side=tk.LEFT)
            save_tag_label2.pack(side=tk.LEFT)
            save_tag_infolabel.pack(side=tk.LEFT)
            # save_tag_dummy_label.pack(side=tk.BOTTOM)
        else:
            save_tag_label.pack_forget()
            save_tag_entry.pack_forget()
            save_tag_label2.pack_forget()
            save_tag_infolabel.pack_forget()

            save_tag_dummy_label.pack()

    def handle_resize(event):
        canvas = event.widget
        canvas_frame = canvas.nametowidget(canvas.itemcget("frame_canvas", "window"))
        min_width = canvas_frame.winfo_reqwidth()
        min_height = canvas_frame.winfo_reqheight()
        canvas.itemconfigure("frame_canvas", width=max(event.width, min_width))
        canvas.itemconfigure("frame_canvas", height=max(event.height, min_height))

        canvas.configure(scrollregion=canvas.bbox("all"))

    # Create the main window
    def _quit():
        root.quit()
        root.destroy()
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", _quit)
    root.geometry("1500x1000")
    bgc = 'gray65'
    buttoncolor = 'gray75'
    root.config(bg=bgc)
    root.option_add("*Background", bgc)

    canvas = tk.Canvas(root, background=bgc)
    frame_canvas = tk.Frame(canvas, background=bgc)

    canvas.pack(expand=True, fill="both")
    canvas.create_window((4, 4), window=frame_canvas, anchor="n", tags="frame_canvas")

    yscrollbar = tk.Scrollbar(canvas, orient="vertical",
                              command=canvas.yview)
    yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    xscrollbar = tk.Scrollbar(canvas, orient="horizontal",
                              command=canvas.xview)
    xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    canvas.bind("<Configure>", handle_resize)

    # canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
    frame_canvas.bind('<Enter>',
                      lambda event: canvas.bind_all("<MouseWheel>",
                                                    lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units")))
    frame_canvas.bind('<Leave>',
                      lambda event: canvas.unbind_all("<MouseWheel>"))

    menu_bar = tk.Menu(root)
    menu_bar.config(bg='lightblue', fg='black')
    # file_menu = tk.Menu(menu_bar)
    file_menu = tk.Menu(menu_bar, tearoff=0, bg="lightblue", fg="black")
    file_menu.add_command(label="New", command=lambda: print("New File"))
    file_menu.add_command(label="Open", command=lambda: print("Open File"))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=lambda: root.destroy())
    menu_bar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menu_bar)

    root.title("LARS Comparison Settings")

    roottop = tk.Frame(frame_canvas)
    roottop.pack(side=tk.TOP)
    rootsettings = tk.Frame(frame_canvas)
    rootsettings.pack(side=tk.TOP)
    rootsubmit = tk.Frame(frame_canvas)
    rootsubmit.pack(side=tk.BOTTOM)

    rootl = tk.Frame(rootsettings)
    rootl.pack(side=tk.LEFT)
    rootr = tk.Frame(rootsettings)
    rootr.pack(side=tk.LEFT)

    # Global info
    running_var = tk.BooleanVar(root, value=False)
    prev_settings_var = tk.Variable(root, value={})
    status_var = tk.StringVar(root, value='nodir')
    default_font_name = tk.font.nametofont('TkTextFont').actual()['family']
    data_dict_var, pair_results_var = CustomVar(), CustomVar()
    data_dict_var.set({})
    pair_results_var.set({})

    common_kwargs = {'update_status': update_status, 'varframe': root}

    # Start building App

    heading('Load Data', frame=roottop, padding=False, side=tk.TOP)
    rootload = tk.Frame(roottop)
    rootload.pack(side=tk.TOP)
    # DIRECTORY
    directory_var, _, _, _, _, _ = labeled_file_select(rootload, headingtxt='Directory',
                                                       subheading="""Select a folder which contains subfolders, each of which contain LARS data in .all format.
All pairs of subfolders will be compared.""",
                                                       label='Enter path to LARS data or select a folder:',
                                                       selection='dir',
                                                       infotext=infotext['directory'], **common_kwargs)
    frame_data_format = tk.Frame(rootload)
    frame_data_format.pack(side=tk.TOP)
    # data_format
    data_format_var, _, _, _, _, _ = labeled_options(frame_data_format, 'Data format to load:',
                                                     padding=padding_setting, vartype=tk.StringVar,
                                                     vardefault='auto', options=['auto', '.all', '.csv', '.tdms', '.npz'],
                                                     infotext=infotext['data_format'], side=tk.LEFT, **common_kwargs)
    # new_data_format
    new_data_format_var, _, _, _, _, _ = labeled_options(frame_data_format, 'Save data to different format (does not work with .all):',
                                                         padding=padding_setting, vartype=tk.StringVar,
                                                         vardefault='none', options=['none', '.csv', '.npz', 'both'],
                                                         infotext=infotext['new_data_format'], **common_kwargs)

    # pickled_data_path

    pickled_data_path_var, _, _, _, _, _ = labeled_file_select(rootload, headingtxt='Pickled Data',
                                                               subheading='Load data from previous analysis.',
                                                               label='Enter path to pickled data or select a file:',
                                                               infotext=infotext['pickled_data_path'], side=tk.LEFT,
                                                               **common_kwargs)
    import_settings_Button = tk.Button(roottop, text="Import Settings", command=import_settings, bg=buttoncolor)
    import_settings_Button.pack(**padding_setting)

    heading('Settings', frame=roottop, side=tk.BOTTOM)

    # DATA DEFINITIONS

    heading("Data", lvl=1, frame=rootl, side=tk.TOP)

    # frange and slc limits
    frame_frange = tk.Frame(rootl)
    frame_frange.pack(**padding_setting, side=tk.TOP)
    frange_min_var, _, _, _, _, _ = labeled_entry(frame_frange, 'Data minimum and maximum frequency:',
                                                  padding=padding_none,
                                                  vardefault=10000, vartype=tk.DoubleVar, side=tk.LEFT,
                                                  infobox=False, **common_kwargs)
    frange_max_var, _, _, _, _, _ = labeled_entry(frame_frange, '-',
                                                  postlabel='Hz', padding=padding_none,
                                                  vardefault=60000, vartype=tk.DoubleVar, side=tk.LEFT,
                                                  infotext=infotext['frange'], **common_kwargs)

    # combine
    combine_var, _, _, _, _, _ = labeled_options(rootl, 'How data within a folder should be combined:',
                                                 padding=padding_setting, vartype=tk.StringVar,
                                                 vardefault='max', options=['max', 'mean'],
                                                 infotext=infotext['combine'], **common_kwargs)

    # grouped_folders
    grouped_folders_var, _, _, _, _, _ = labeled_options(rootr, 'Use grouped folder structure:', padding=padding_setting,
                                                         vartype=tk.StringVar, vardefault=bool_options[1],
                                                         infotext=infotext['grouped_folders'], **common_kwargs)

    # part_matching
    frame_part_matching = tk.Frame(rootr)
    frame_part_matching.pack(**padding_setting)

    part_matching_text_var = tk.StringVar(root, value='')
    part_matching_strategy_var = tk.StringVar(root, value='list')

    part_matching_Button = tk.Button(rootr, text="Define Known Part Matching", bg='gray75',
                                     command=lambda: part_matching_text_var.set(open_part_matching_window(root, grouped_folders_var, part_matching_text_var, part_matching_strategy_var, **common_kwargs)))
    part_matching_Button.pack()

    # PLOTTING AND PRINTING

    # plot, plot_detail, plot_recursive_noise and plot_classification
    frame_plot = tk.Frame(rootl)
    frame_plot.pack(**padding_option, side=tk.TOP)

    heading("Plotting and Printing", lvl=1, frame=frame_plot, side=tk.TOP)

    frame_plot_label = tk.Frame(frame_plot)
    frame_plot_label.pack(side=tk.TOP)

    plot_label = tk.Label(
        frame_plot_label, text="Create plots (with fitting details)\n(of recursive noise iterations) (of classification):")
    plot_label.pack(side=tk.BOTTOM)

    frame_plot_menus = tk.Frame(frame_plot)
    frame_plot_menus.pack(side=tk.BOTTOM)

    plot_var = tk.StringVar(root, value=bool_options[1])
    plot_detail_var = tk.StringVar(root, value=bool_options[1])
    plot_recursive_noise_var = tk.StringVar(root, value=bool_options[1])
    plot_classification_var = tk.StringVar(root, value=bool_options[1])

    plot_var.trace_add("write", update_status)
    plot_detail_var.trace_add("write", update_status)
    plot_recursive_noise_var.trace_add("write", update_status)
    plot_classification_var.trace_add("write", update_status)

    plot_menu = tk.OptionMenu(frame_plot_menus, plot_var, *bool_options)
    plot_detail_menu = tk.OptionMenu(frame_plot_menus, plot_detail_var, *bool_options)
    plot_recursive_noise_menu = tk.OptionMenu(frame_plot_menus, plot_recursive_noise_var, *bool_options)
    plot_classification_menu = tk.OptionMenu(frame_plot_menus, plot_classification_var, *bool_options)

    plot_menu.config(bg='gray75', highlightthickness=0)
    plot_detail_menu.config(bg='gray75', highlightthickness=0)
    plot_recursive_noise_menu.config(bg='gray75', highlightthickness=0)
    plot_classification_menu.config(bg='gray75', highlightthickness=0)

    plot_menu.grid(row=0, column=0)
    plot_detail_menu.grid(row=0, column=1)
    plot_recursive_noise_menu.grid(row=1, column=0)
    plot_classification_menu.grid(row=1, column=1)

    # peak_plot_width
    peak_plot_width_var, _, _, _, _, _ = labeled_entry(rootl, 'Width of peak fit plots:',
                                                       postlabel='kHz', padding=padding_setting,
                                                       vardefault=20, vartype=tk.DoubleVar,
                                                       infotext=infotext['peak_plot_width'], **common_kwargs)

    # show_plots
    frame_show_save_plots = tk.Frame(rootl)
    frame_show_save_plots.pack(side=tk.TOP)
    show_plots_var, _, _, _, _, _ = labeled_options(frame_show_save_plots, 'Show plots:', padding=padding_setting,
                                                    side=tk.LEFT, vartype=tk.StringVar, vardefault=bool_options[1],
                                                    infotext=infotext['show_plots'], **common_kwargs)

    # save_plots
    save_plots_var, _, _, _, _, _ = labeled_options(frame_show_save_plots, 'Save plots:', padding=padding_setting,
                                                    side=tk.LEFT, vartype=tk.StringVar, vardefault=bool_options[1],
                                                    infotext=infotext['save_plots'], **common_kwargs)

    # PRINT_MODE
    PRINT_MODE_var, _, _, _, _, _ = labeled_options(rootl, 'Print details:',
                                                    padding=padding_setting, vartype=tk.StringVar,
                                                    vardefault='sparse', options=['none', 'sparse', 'full'],
                                                    infotext=infotext['PRINT_MODE'], **common_kwargs)
    # SAVING

    heading("Saving", lvl=1, frame=rootl, padding=False)

    # save_data
    save_data_var, _, _, _, _, _ = labeled_options(rootl, 'Save data to .pkl file:', padding=padding_setting,
                                                   vartype=tk.StringVar, vardefault=bool_options[1], command=hide_save_tag,
                                                   infotext=infotext['save_data'], **common_kwargs)

    # save_results
    save_results_var, _, _, _, _, _ = labeled_options(rootl, 'Save results to .pkl file:', padding=padding_setting,
                                                      vartype=tk.StringVar, vardefault=bool_options[0], command=hide_save_tag,
                                                      infotext=infotext['save_results'], **common_kwargs)

    # save_tag
    save_tag_var, frame_save_tag, save_tag_label, save_tag_entry, save_tag_label2, save_tag_infolabel =\
        labeled_entry(rootl, 'Save filename: data_dict... and peak_results', postlabel='.pkl', padding=padding_setting,
                      vardefault='', vartype=tk.StringVar,
                      infotext=infotext['save_tag'], **common_kwargs)
    save_tag_var.trace_add("write", update_save_tag_label)

    save_tag_dummy_label = tk.Label(frame_save_tag, text=" "*42, font=("Courier New", 9))

    # save_directory
    save_directory_var, _, _, _, _, _ = labeled_file_select(rootl, subheading='Enter path or select a folder to save results, data, and/or plots:',
                                                            selection='dir', vardefault='Same as LARS Data Directory',
                                                            infotext=infotext['save_folder'], **common_kwargs)

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
    baseline_smoothness_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline smoothness: 10^',
                                                           padding=padding_setting, vardefault=12, vartype=tk.DoubleVar,
                                                           infotext=infotext['baseline_smoothness'], **common_kwargs)

    # baseline_polyorder
    baseline_polyorder_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline polyorder:',
                                                          padding=padding_setting, vardefault=2, vartype=tk.IntVar,
                                                          infotext=infotext['baseline_polyorder'], **common_kwargs)

    # baseline_itermax
    baseline_itermax_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline itermax:',
                                                        padding=padding_setting, vardefault=10, vartype=tk.IntVar,
                                                        infotext=infotext['baseline_itermax'], **common_kwargs)

    # sgf_windowsize
    sgf_windowsize_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF Windowsize:',
                                                      padding=padding_setting, vardefault=101, vartype=tk.IntVar,
                                                      infotext=infotext['sgf_windowsize'], **common_kwargs)

    # sgf_applications
    sgf_applications_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF Applications:',
                                                        padding=padding_setting, vardefault=2, vartype=tk.IntVar,
                                                        infotext=infotext['sgf_applications'], **common_kwargs)

    # sgf_polyorder
    sgf_polyorder_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF polyorder:',
                                                     padding=padding_setting, vardefault=0, vartype=tk.IntVar,
                                                     infotext=infotext['sgf_polyorder'], **common_kwargs)

    # headings
    heading("Peak Finding", lvl=2, frame=frame_peak_fitl, padding=False)
    heading("Noise Reduction", lvl=2, frame=frame_peak_fitr, padding=False)

    # peak_height_min
    peak_height_min_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak height minimum: noise *',
                                                       padding=padding_setting, vardefault=0.2, vartype=tk.DoubleVar,
                                                       infotext=infotext['peak_height_min'], **common_kwargs)

    # peak_prominence_min
    peak_prominence_min_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak prominence minimum: noise *',
                                                           padding=padding_setting, vardefault=0.2, vartype=tk.DoubleVar,
                                                           infotext=infotext['peak_prominence_min'], **common_kwargs)

    # peak_ph_ratio_min
    peak_ph_ratio_min_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak prominence-to-height minimum:',
                                                         padding=padding_setting, vardefault=0.5, vartype=tk.DoubleVar,
                                                         infotext=infotext['peak_ph_ratio_min'], **common_kwargs)

    # recursive_noise_reduction
    recursive_noise_reduction_var, _, _, _, _, _ = labeled_options(frame_peak_fitr, 'Recursively reduce noise:',
                                                                   padding=padding_setting, vartype=tk.StringVar,
                                                                   vardefault=bool_options[0],
                                                                   infotext=infotext['recursive_noise_reduction'], **common_kwargs)

    # max_noise_reduction_iter
    max_noise_reduction_iter_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'Max noise reduction iterations:',
                                                                padding=padding_setting, vardefault=10, vartype=tk.IntVar,
                                                                infotext=infotext['max_noise_reduction_iter'], **common_kwargs)

    # regularization_ratio
    regularization_ratio_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'Noise reduction regularization factor:',
                                                            padding=padding_setting, vardefault=0.5, vartype=tk.DoubleVar,
                                                            infotext=infotext['regularization_ratio'], **common_kwargs)

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
    max_stretch_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Max stretching: 1 ±',
                                                   padding=padding_setting, vardefault=0.02, vartype=tk.DoubleVar,
                                                   infotext=infotext['max_stretch'], **common_kwargs)

    # num_stretches
    num_stretches_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretches per iteration:',
                                                     padding=padding_setting, vardefault=1000, vartype=tk.IntVar,
                                                     infotext=infotext['num_stretches'], **common_kwargs)

    # stretching_iterations
    stretching_iterations_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretching iterations:',
                                                             padding=padding_setting, vardefault=10, vartype=tk.IntVar,
                                                             infotext=infotext['stretching_iterations'], **common_kwargs)

    # stretch_iteration_factor
    stretch_iteration_factor_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl,
                                                                'Factor to reduce stretch space each iteration:',
                                                                padding=padding_setting, vardefault=5,
                                                                vartype=tk.DoubleVar,
                                                                infotext=infotext['stretch_iteration_factor'], **common_kwargs)

    # nw_normalized
    nw_normalized_var, _, _, _, _, _ = labeled_options(frame_peak_matchr, 'Normalize frequency for peak matching:',
                                                       padding=padding_setting, vartype=tk.StringVar,
                                                       vardefault=bool_options[1],
                                                       command=update_peak_match_window_label,
                                                       infotext=infotext['nw_normalized'], **common_kwargs)

    # peak_match_window
    # var, frame, label1, entry, label2, infolabel
    peak_match_window_var, _, _, _, peak_match_window_label2, peak_match_window_infobox =\
        labeled_entry(frame_peak_matchr,
                      'Max matching difference:',
                      postlabel='Hz',
                      padding=padding_setting, vardefault=150,
                      vartype=tk.DoubleVar,
                      infotext=infotext['peak_match_window'], **common_kwargs)

    # matching_penalty_order
    matching_penalty_order_var, _, _, _, _, _ = labeled_entry(frame_peak_matchr,
                                                              'Matching penalty order:',
                                                              padding=padding_setting, vardefault=1, vartype=tk.DoubleVar,
                                                              infotext=infotext['matching_penalty_order'], **common_kwargs)

    # dummy
    frame_dummy = tk.Frame(frame_peak_matchr)
    frame_dummy.pack(**padding_setting, side=tk.TOP)
    dummy_label = tk.Label(frame_dummy, text="")
    dummy_label.pack(side=tk.LEFT)

    # SUBMIT
    frame_submit = tk.Frame(rootsubmit)
    frame_submit.pack(**padding_heading, side=tk.BOTTOM)

    submit_Button = tk.Button(frame_canvas, text="Run Code", bg='firebrick4', fg='white',
                              width=20, height=2, command=submit, font=(default_font_name, 20, "bold"))
    submit_Button.pack(**padding_heading)

    frame_status = tk.Frame(rootsubmit)
    frame_status.pack(side=tk.BOTTOM)

    plot_Button = tk.Button(frame_status, text="View Plots",
                            command=lambda: open_plot_window(root, data_dict_var,
                                                             pair_results_var,
                                                             frange_min_var,
                                                             frange_max_var),
                            bg=buttoncolor)
    plot_Button.pack(**padding_heading, side=tk.LEFT)

    status_label0 = tk.Label(frame_status, text='Status:', font=(default_font_name, 12))
    status_label0.pack(**padding_setting, side=tk.LEFT)
    status_label = tk.Label(frame_status, text='No directory selected.',
                            bg='firebrick4', fg='white', font=(default_font_name, 12))
    status_label.pack(**padding_setting, side=tk.RIGHT)

    # Start the main loop
    root.mainloop()


if __name__ == '__main__':
    run_app()
