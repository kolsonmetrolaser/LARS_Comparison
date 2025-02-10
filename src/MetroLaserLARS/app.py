# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# External imports
import tkinter as tk
import pickle
import sys
from shutil import move as shmove
import os.path as osp
import tempfile
# import threading
import queue
# import traceback
from numpy import log10
from PyThreadKiller import PyThreadKiller

# Internal imports
try:
    import infotext as it
    from LARS_Comparison import LARS_Comparison_from_app
    from app_helpers import heading, labeled_options, labeled_file_select, labeled_entry
    from app_helpers import padding_none, padding_setting, padding_option, padding_heading
    from app_helpers import bool_options, CustomVar, make_button, log_decorator, open_log_window
    from app_helpers import button_color, active_bg, active_fg, icon_ML, open_progress_window
    from app_helpers import background_color as bgc
    from app_window_part_matching import open_part_matching_window
    from app_window_plot import open_plot_window
    from app_window_clustering import open_clustering_window
    from app_window_results_table import open_results_table_window
    from app_window_peak_list import open_peak_list_window
except ModuleNotFoundError:
    import MetroLaserLARS.infotext as it  # type: ignore
    from MetroLaserLARS.LARS_Comparison import LARS_Comparison_from_app  # type: ignore
    from MetroLaserLARS.app_helpers import heading, labeled_options, labeled_file_select, labeled_entry  # type: ignore
    from MetroLaserLARS.app_helpers import padding_none, padding_setting, padding_option, padding_heading  # type: ignore
    from MetroLaserLARS.app_helpers import bool_options, CustomVar, make_button, log_decorator, open_log_window  # type: ignore
    from MetroLaserLARS.app_helpers import button_color, active_bg, active_fg, icon_ML, open_progress_window  # type: ignore
    from MetroLaserLARS.app_helpers import background_color as bgc  # type: ignore
    from MetroLaserLARS.app_window_part_matching import open_part_matching_window  # type: ignore
    from MetroLaserLARS.app_window_plot import open_plot_window  # type: ignore
    from MetroLaserLARS.app_window_clustering import open_clustering_window  # type: ignore
    from MetroLaserLARS.app_window_results_table import open_results_table_window  # type: ignore
    from MetroLaserLARS.app_window_peak_list import open_peak_list_window  # type: ignore


def run_app_main():
    def make_settings(suppress=False):
        try:
            settings = {}
            # fmt: off
            # DATA
            settings['directory']                 = directory_var.get() # noqa
            settings['data_format']               = data_format_var.get() # noqa
            settings['new_data_format']           = new_data_format_var.get() # noqa
            settings['pickled_data_path']         = pickled_data_path_var.get() # noqa
            settings['interpolate_raw_spectra']   = True # noqa
            # DATA DEFINITIONS
            settings['stft']                      =True if stft_var.get() == 'True' else False # noqa
            settings['slc_limits']                = (slc_limits_min_var.get(), slc_limits_max_var.get()) # noqa
            settings['combine']                   = combine_var.get() # noqa
            settings['grouped_folders']           = True if grouped_folders_var.get() == 'True' else False # noqa
            settings['part_matching_text']        = part_matching_text_var.get() # noqa
            settings['part_matching_strategy']    = part_matching_strategy_var.get() # noqa
            settings['reference']                 = '' if reference_var.get()=='No Reference' else reference_var.get() # noqa
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
            settings['peak_fitting_strategy']     = peak_fitting_strategy_var.get() # noqa
            # baseline removal
            settings['baseline_smoothness']       = 10**baseline_smoothness_var.get() # noqa
            settings['baseline_polyorder']        = baseline_polyorder_var.get() # noqa
            settings['baseline_itermax']          = baseline_itermax_var.get() # noqa
            # smoothing
            settings['sgf_applications']          = sgf_applications_var.get() # noqa
            settings['sgf_windowsize']            = sgf_windowsize_var.get() # noqa
            settings['sgf_polyorder']             = sgf_polyorder_var.get() # noqa
            settings['hybrid_smoothing']          = True if hybrid_smoothing_var.get() == 'True' else False # noqa
            # peak finding
            settings['peak_height_min']           = peak_height_min_var.get()-1 # noqa
            settings['peak_prominence_min']       = peak_prominence_min_var.get()-1 # noqa
            settings['peak_ph_ratio_min']         = peak_ph_ratio_min_var.get() # noqa
            # noise reduction
            settings['recursive_noise_reduction'] = True if recursive_noise_reduction_var.get() == 'True' else False  # noqa
            settings['max_noise_reduction_iter']  = max_noise_reduction_iter_var.get() # noqa
            settings['regularization_ratio']      = regularization_ratio_var.get() # noqa
            # machine learning
            settings['ml_threshold']              = ml_threshold_var.get() # noqa
            settings['ml_weights_path']           = ml_weights_path_var.get() # noqa
            # PEAK MATCHING
            # stretching
            settings['max_stretch']               = max_stretch_var.get() # noqa
            settings['num_stretches']             = int(10**num_stretches_var.get()) # noqa
            # settings['stretching_iterations']     = stretching_iterations_var.get() # noqa
            # settings['stretch_iteration_factor']  = stretch_iteration_factor_var.get() # noqa
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
            settings['app_version']               = (0, 2, 2)  # noqa

            try:
                settings['progress_bars']         = progress_vars.get() # noqa
            except NameError:
                settings['progress_bars']         = None  # noqa

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
            stft_var.set(                      ('True' if settings['stft'] else 'False') if 'stft' in settings else 'False') # noqa
            slc_limits_min_var.set(            settings['slc_limits'][0] if 'slc_limits' in settings else 10000) # noqa
            slc_limits_max_var.set(            settings['slc_limits'][1] if 'slc_limits' in settings else 60000) # noqa
            combine_var.set(                   settings['combine'] if 'combine' in settings else 'max') # noqa
            grouped_folders_var.set(           ('True' if settings['grouped_folders'] else 'False') if 'grouped_folders' in settings else 'False') # noqa
            part_matching_text_var.set(        settings['part_matching_text'] if 'part_matching_text' in settings else '') # noqa
            part_matching_strategy_var.set(    settings['part_matching_strategy'] if 'part_matching_strategy' in settings else 'list') # noqa
            reference_var.set(                 settings['reference'] if 'reference' in settings else 'No Reference') # noqa
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
            peak_fitting_strategy_var.set(     settings['peak_fitting_strategy'] if 'peak_fitting_strategy' in settings else 'Standard') # noqa
            # baseline removal
            baseline_smoothness_var.set(       log10(settings['baseline_smoothness']) if 'baseline_smoothness' in settings else 12) # noqa
            baseline_polyorder_var.set(        settings['baseline_polyorder'] if 'baseline_polyorder' in settings else 2) # noqa
            baseline_itermax_var.set(          settings['baseline_itermax'] if 'baseline_itermax' in settings else 10) # noqa
            # smoothing
            sgf_applications_var.set(          settings['sgf_applications'] if 'sgf_applications' in settings else 2) # noqa
            sgf_windowsize_var.set(            settings['sgf_windowsize'] if 'sgf_windowsize' in settings else 101) # noqa
            sgf_polyorder_var.set(             settings['sgf_polyorder'] if 'sgf_polyorder' in settings else 0) # noqa
            hybrid_smoothing_var.set(          ('True' if settings['hybrid_smoothing'] else 'False') if 'hybrid_smoothing' in settings else 'False') # noqa
            # peak finding
            peak_height_min_var.set(           settings['peak_height_min']+1 if 'peak_height_min' in settings else 0.2) # noqa
            peak_prominence_min_var.set(       settings['peak_prominence_min']+1 if 'peak_prominence_min' in settings else 0.2) # noqa
            peak_ph_ratio_min_var.set(         settings['peak_ph_ratio_min'] if 'peak_ph_ratio_min' in settings else 0.5) # noqa
            # noise reduction
            recursive_noise_reduction_var.set( ('True' if settings['recursive_noise_reduction'] else 'False') if 'recursive_noise_reduction' in settings else 'True') # noqa
            max_noise_reduction_iter_var.set(  settings['max_noise_reduction_iter'] if 'max_noise_reduction_iter' in settings else 10) # noqa
            regularization_ratio_var.set(      settings['regularization_ratio'] if 'regularization_ratio' in settings else 0.5) # noqa
            # machine learning
            ml_threshold_var.set(              settings['ml_threshold'] if 'ml_threshold' in settings else 0.01) # noqa
            ml_weights_path_var.set(           settings['ml_weights_path'] if 'ml_weights_path' in settings else 'default weights') # noqa
            # PEAK MATCHING
            # stretching
            max_stretch_var.set(               settings['max_stretch'] if 'max_stretch' in settings else 0.02) # noqa
            num_stretches_var.set(             log10(settings['num_stretches']) if 'num_stretches' in settings else 4) # noqa
            # stretching_iterations_var.set(     settings['stretching_iterations'] if 'stretching_iterations' in settings else 10) # noqa
            # stretch_iteration_factor_var.set(  settings['stretch_iteration_factor'] if 'stretch_iteration_factor' in settings else 5) # noqa
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

    def update_status(*_):
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
        else:
            bg, fg = 'white', 'black'
            label_text = ''
            label_state = tk.NORMAL

        submit_Button.config(bg=bg, fg=fg, text=tk.Button_text)
        status_label.config(bg=bg, fg=fg, text=label_text, state=label_state)

        root.update()
        return

    def submit():

        if status_var.get() in ['nodir', 'running']:
            return

        running_var.set(True)
        update_status()

        progress_vars.set([tk.DoubleVar(root, value=0), tk.DoubleVar(root, value=0)])
        progress_texts = ['Loading data...', 'Analyzing data...']
        progress_window = open_progress_window(root, progress_vars.get(), progress_texts, status_label)

        settings = make_settings()

        def progress_window_cleanup():
            running_var.set(False)
            with open(log_file_loc_var.get(), 'a', encoding="utf-8") as f:
                f.write(log_var.get())
            log_var.set('')
            update_status()
            progress_window.destroy()

        if settings['PRINT_MODE'] in ['full']:
            print(settings)

        # Unthreaded
        if not threaded_var.get():
            result = LARS_Comparison_from_app(settings)
            if result[0] == -1:
                e, tb = result[1]
                print('Error in main analysis code')
                print(tb)
                tk.messagebox.showerror('Error',
                                        f"""Error in analysis, exiting. Error:
{e}

See the log for more detail, available in the log window or
{log_file_loc_var.get()}""")
                progress_window_cleanup()
                return
            else:
                data_dict_var.set(result[0])
                pair_results_var.set(result[1])
                prev_settings_var.set(settings)
                progress_window_cleanup()
        else:
            # Threaded

            def LARS_comparison_worker(q, settings):
                print('starting main code')
                q.put(LARS_Comparison_from_app(settings))

            result_queue = queue.Queue()
            thread = PyThreadKiller(target=LARS_comparison_worker, args=(result_queue, settings))
            thread.daemon = True
            print('starting thread')
            thread.start()

            def on_progress_window_closing():
                if tk.messagebox.askokcancel("Stop code?", "Closing this window will stop the analysis code. Are you sure?", parent=progress_window):
                    thread.kill()
                    progress_window_cleanup()
                    print("""
        ==============================
        User forcefully stopped thread
        ==============================
        """)
                return

            progress_window.protocol("WM_DELETE_WINDOW", on_progress_window_closing)

            def check_result():
                try:
                    result = result_queue.get(block=False)
                    if result[0] == -1:
                        e, tb = result[1]
                        print('Error in main analysis code')
                        print(tb)
                        tk.messagebox.showerror('Error',
                                                f"""Error in analysis, exiting. Error:
    {e}

    See the log for more detail, available in the log window or
    {log_file_loc_var.get()}""")
                        progress_window_cleanup()

                        return
                    print('thread finished and result read')
                    data_dict, pair_results = result
                    data_dict_var.set(data_dict)
                    pair_results_var.set(pair_results)

                    prev_settings_var.set(settings)
                    progress_window_cleanup()

                    root.after(100, check_result)
                except queue.Empty:
                    root.after(100, check_result)

            root.after(100, check_result)

        return

    def import_pickle(mode, *_, **kwargs):
        paths = []
        if mode == 'settings':
            paths.append(tk.filedialog.askopenfilename(**kwargs, title='Choose settings to import',
                                                       filetypes=(('Settings Files', '*settings*.pkl'),
                                                                  ('All Files', '*'))))
        elif mode == 'data':
            paths.append(tk.filedialog.askopenfilename(**kwargs, title='Choose data dict to import',
                                                       filetypes=(('Pickled Data Dictionaries', '*data_dict*.pkl'),
                                                                  ('All Files', '*'))))
            if paths[0]:
                paths.append(tk.filedialog.askopenfilename(**kwargs, title='Choose pair results to import',
                                                           filetypes=(('Pickled Pair Results', '*pair_results*.pkl'),
                                                                      ('All Files', '*'))))

        data = []
        for path in [p for p in paths if p]:
            try:
                with open(path, 'rb') as f:
                    data.append(pickle.load(f))
            except Exception:
                tk.messagebox.showerror(title='Load Failed', message=f'{mode} import failed.')
                return
        if data:
            if mode == 'settings':
                set_settings(data[0])
            elif mode == 'data' and len(data) == 2:
                data_dict_var.set(data[0])
                pair_results_var.set(data[1])
            elif mode == 'data':
                tk.messagebox.showerror(title='Load Failed', message=f'{mode} import failed.')
                return
        return

    def update_peak_match_window_label(entry_text):
        # Update the label text with the selected option
        peak_match_window_infobox.pack_forget()
        txt = "Hz" if entry_text == 'False' else '*peak_position/Hz'
        peak_match_window_label2.config(text=txt)
        peak_match_window_infobox.pack(side=tk.LEFT)

    def update_save_tag_label(*_):
        # Update the label text with the selected option
        txt = "Save filename: data_dict... and peak_results" if save_tag_var.get(
        ) == '' else 'Save filename: data_dict_... and peak_results_'
        save_tag_label.config(text=txt)

    def hide_save_tag(*_):
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
        for t in log_var.trace_info():
            log_var.trace_remove(*t)
        root.quit()
        root.destroy()
    root = tk.Tk()

    root.protocol("WM_DELETE_WINDOW", _quit)
    root.config(bg=bgc)
    root.option_add("*Background", bgc)
    root.wm_iconphoto(False, tk.PhotoImage(file=icon_ML))

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

    canvas.config(xscrollcommand=xscrollbar.set)
    canvas.config(yscrollcommand=yscrollbar.set)

    canvas.bind("<Configure>", handle_resize)

    # canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
    canvas.bind('<Enter>',
                lambda event: canvas.bind_all("<MouseWheel>",
                                              lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units")))
    canvas.bind('<Leave>',
                lambda event: canvas.unbind_all("<MouseWheel>"))

    # root.geometry("1600x900")
    # maximize window
    root.state("zoomed")

    # Needed for window to appear on top of others in some cases
    root.wm_attributes('-topmost', True)
    root.wm_attributes('-topmost', False)

    # menu_bar = tk.Menu(root, background=bgc)
    # menu_bar.config(bg=bgc, fg='black')
    # # file_menu = tk.Menu(menu_bar)
    # file_menu = tk.Menu(menu_bar, tearoff=0, bg=bgc, fg="black")
    # file_menu.add_command(label="New", command=lambda: print("New File"))
    # file_menu.add_command(label="Open", command=lambda: print("Open File"))
    # file_menu.add_separator()
    # file_menu.add_command(label="Exit", command=_quit)
    # menu_bar.add_cascade(label="File", menu=file_menu)
    # root.config(menu=menu_bar)

    root.title("LARS Comparison Settings")

    roottop = tk.Frame(frame_canvas)
    roottop.pack(side=tk.TOP)
    rootsettings = tk.Frame(frame_canvas)
    rootsettings.pack(side=tk.TOP)
    rootsubmit = tk.Frame(frame_canvas)
    rootsubmit.pack(side=tk.TOP)
    rootbuttons = tk.Frame(frame_canvas)
    rootbuttons.pack(side=tk.TOP)

    rootl = tk.Frame(rootsettings)
    rootl.pack(side=tk.LEFT)
    rootr = tk.Frame(rootsettings)
    rootr.pack(side=tk.LEFT)

    # Global info
    using_temp_file = tk.BooleanVar(root, value=True)
    running_var = tk.BooleanVar(root, value=False)
    prev_settings_var = tk.Variable(root, value={})
    status_var = tk.StringVar(root, value='nodir')
    default_font_name = tk.font.nametofont('TkTextFont').actual()['family']
    data_dict_var, pair_results_var, progress_vars = CustomVar(), CustomVar(), CustomVar()
    data_dict_var.set({})
    pair_results_var.set({})
    progress_vars.set(None)
    option_menu_kwargs = {'bg': button_color, 'highlightthickness': 0,
                          'activebackground': active_bg, 'activeforeground': active_fg}

    common_kwargs = {'update_status': update_status, 'varframe': root}
    threaded_var = tk.BooleanVar(root, value=True)

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
                                                       infotext=it.directory, **common_kwargs)

    frame_data_format = tk.Frame(rootload)
    frame_data_format.pack(side=tk.TOP)
    # data_format
    data_format_var, _, _, _, _, _ = labeled_options(frame_data_format, 'Data format to load:',
                                                     padding=padding_setting, vartype=tk.StringVar,
                                                     vardefault='auto', options=['auto', '.all', '.csv', '.tdms', '.npz'],
                                                     infotext=it.data_format, side=tk.LEFT, **common_kwargs)
    # new_data_format
    new_data_format_var, _, _, _, _, _ = labeled_options(frame_data_format, 'Save data to different format (does not work when loading .all):',
                                                         padding=padding_setting, vartype=tk.StringVar,
                                                         vardefault='none',
                                                         options=['none', '.csv', '.npz', '.all', '.csv and .npz',
                                                                  '.all and .npz', 'all of the above'],
                                                         infotext=it.new_data_format, **common_kwargs)

    # pickled_data_path

    pickled_data_path_var, _, _, _, _, _ = labeled_file_select(rootload, headingtxt='Pickled Data',
                                                               subheading='Load data from previous analysis.',
                                                               label='Enter path to pickled data or select a file:',
                                                               infotext=it.pickled_data_path, side=tk.LEFT,
                                                               filetype='pickle',
                                                               **common_kwargs)
    frame_load_button = tk.Frame(roottop)
    frame_load_button.pack(side=tk.TOP)
    make_button(frame_load_button, text="Import Settings",
                command=lambda: import_pickle('settings'), side=tk.LEFT)
    make_button(frame_load_button, text="Import Data",
                command=lambda: import_pickle('data'), side=tk.LEFT)

    heading('Settings', frame=roottop, side=tk.BOTTOM)

    # DATA DEFINITIONS

    heading("Data", lvl=1, frame=rootl, side=tk.TOP)

    # stft
    stft_var, _, _, _, _, _ = labeled_options(rootl, 'Use STFT:', padding=padding_setting,
                                              side=tk.LEFT, vartype=tk.StringVar, vardefault=bool_options[1],
                                              infotext=it.stft, **common_kwargs)

    # slc_limits and slc limits
    frame_slc_limits = tk.Frame(rootl)
    frame_slc_limits.pack(**padding_setting, side=tk.TOP)
    slc_limits_min_var, _, _, _, _, _ = labeled_entry(frame_slc_limits, 'Data minimum and maximum frequency:',
                                                      padding=padding_none,
                                                      vardefault=10000, vartype=tk.DoubleVar, side=tk.LEFT,
                                                      infobox=False, **common_kwargs)
    slc_limits_max_var, _, _, _, _, _ = labeled_entry(frame_slc_limits, '-',
                                                      postlabel='Hz', padding=padding_none,
                                                      vardefault=60000, vartype=tk.DoubleVar, side=tk.LEFT,
                                                      infotext=it.slc_limits, **common_kwargs)

    # combine
    combine_var, _, _, _, _, _ = labeled_options(rootl, 'How data within a folder should be combined:',
                                                 padding=padding_setting, vartype=tk.StringVar,
                                                 vardefault='max', options=['max', 'mean', 'none', 'all'],
                                                 infotext=it.combine, **common_kwargs)

    # grouped_folders
    grouped_folders_var, _, _, _, _, _ = labeled_options(rootr, 'Use grouped folder structure:', padding=padding_setting,
                                                         vartype=tk.StringVar, vardefault=bool_options[1],
                                                         infotext=it.grouped_folders, **common_kwargs)

    # part_matching
    frame_part_matching = tk.Frame(rootr)
    frame_part_matching.pack(**padding_setting)

    part_matching_text_var = tk.StringVar(root, value='')
    part_matching_strategy_var = tk.StringVar(root, value='list')
    reference_var = tk.StringVar(root, value='No Reference')

    make_button(rootr, text="Define Known Part Matching",
                command=lambda: part_matching_text_var.set(
                    open_part_matching_window(
                        root,
                        grouped_folders_var,
                        part_matching_text_var,
                        part_matching_strategy_var,
                        reference_var,
                        **common_kwargs)
                ), side=tk.TOP
                )
    part_matching_text_var.trace_add("write", update_status)
    part_matching_strategy_var.trace_add("write", update_status)

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

    plot_menu.config(**option_menu_kwargs)
    plot_detail_menu.config(**option_menu_kwargs)
    plot_recursive_noise_menu.config(**option_menu_kwargs)
    plot_classification_menu.config(**option_menu_kwargs)

    plot_menu.grid(row=0, column=0)
    plot_detail_menu.grid(row=0, column=1)
    plot_recursive_noise_menu.grid(row=1, column=0)
    plot_classification_menu.grid(row=1, column=1)

    # peak_plot_width
    peak_plot_width_var, _, _, _, _, _ = labeled_entry(rootl, 'Width of peak fit plots:',
                                                       postlabel='kHz', padding=padding_setting,
                                                       vardefault=20, vartype=tk.DoubleVar,
                                                       infotext=it.peak_plot_width, **common_kwargs)

    # show_plots
    frame_show_save_plots = tk.Frame(rootl)
    frame_show_save_plots.pack(side=tk.TOP)
    show_plots_var, _, _, _, _, _ = labeled_options(frame_show_save_plots, 'Show plots:', padding=padding_setting,
                                                    side=tk.LEFT, vartype=tk.StringVar, vardefault=bool_options[1],
                                                    infotext=it.show_plots, **common_kwargs)

    # save_plots
    save_plots_var, _, _, _, _, _ = labeled_options(frame_show_save_plots, 'Save plots:', padding=padding_setting,
                                                    side=tk.LEFT, vartype=tk.StringVar, vardefault=bool_options[1],
                                                    infotext=it.save_plots, **common_kwargs)

    # PRINT_MODE
    PRINT_MODE_var, _, _, _, _, _ = labeled_options(rootl, 'Print details:',
                                                    padding=padding_setting, vartype=tk.StringVar,
                                                    vardefault='full', options=['none', 'sparse', 'full'],
                                                    infotext=it.PRINT_MODE, **common_kwargs)
    # SAVING

    heading("Saving", lvl=1, frame=rootl, padding=False)

    # save_data
    save_data_var, _, _, _, _, _ = labeled_options(rootl, 'Save data to .pkl file:', padding=padding_setting,
                                                   vartype=tk.StringVar, vardefault=bool_options[0], command=hide_save_tag,
                                                   infotext=it.save_data, **common_kwargs)

    # save_results
    save_results_var, _, _, _, _, _ = labeled_options(rootl, 'Save results to .pkl file:', padding=padding_setting,
                                                      vartype=tk.StringVar, vardefault=bool_options[0], command=hide_save_tag,
                                                      infotext=it.save_results, **common_kwargs)

    # save_tag
    save_tag_var, frame_save_tag, save_tag_label, save_tag_entry, save_tag_label2, save_tag_infolabel =\
        labeled_entry(rootl, 'Save filename: data_dict... and peak_results', postlabel='.pkl', padding=padding_setting,
                      vardefault='', vartype=tk.StringVar,
                      infotext=it.save_tag, **common_kwargs)
    save_tag_var.trace_add("write", update_save_tag_label)

    save_tag_dummy_label = tk.Label(frame_save_tag, text=" "*42, font=("Courier New", 9))

    # save_directory
    save_directory_var, _, _, _, _, _ = labeled_file_select(rootl, subheading='Enter path or select a folder to save results, data, and/or plots:',
                                                            selection='dir', vardefault='Same as LARS Data Directory',
                                                            infotext=it.save_folder, **common_kwargs)

    def update_peak_fitting_settings(*args):
        if peak_fitting_strategy_var.get() == 'Machine Learning':
            for o in peak_fitting_objects_std:
                o.pack_forget()
            for o in peak_fitting_objects_ml:
                o.pack()
        else:
            for o in peak_fitting_objects_ml:
                o.pack_forget()
            for o in peak_fitting_objects_std:
                o.pack()
        return

    # PEAK FITTING
    frame_peak_fit = tk.Frame(rootr)
    frame_peak_fit.pack(side=tk.TOP)
    frame_peak_fit_split = tk.Frame(frame_peak_fit)
    frame_peak_fit_split.pack(side=tk.BOTTOM)
    heading("Peak Fitting", lvl=1, frame=frame_peak_fit, padding=False)

    peak_fitting_strategy_var, _, _, _, _, _ = labeled_options(frame_peak_fit, 'Peak Fitting Algorithm:',
                                                               padding=padding_setting, vartype=tk.StringVar,
                                                               vardefault='Standard',
                                                               options=['Standard', 'Machine Learning'],
                                                               infotext=it.peak_fitting_strategy, side=tk.TOP,
                                                               **common_kwargs)
    peak_fitting_strategy_var.trace_add("write", update_peak_fitting_settings)

    frame_peak_fitl = tk.Frame(frame_peak_fit_split)
    frame_peak_fitl.pack(side=tk.LEFT)
    frame_peak_fitr = tk.Frame(frame_peak_fit_split)
    frame_peak_fitr.pack(side=tk.LEFT)
    heading_baseline = heading("Baseline", lvl=2, frame=frame_peak_fitl, padding=False)
    heading_smoothing = heading("Smoothing", lvl=2, frame=frame_peak_fitr, padding=False)

    # baseline_smoothness
    baseline_smoothness_var, baseline_smoothness_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'Basline smoothness: 10^',
                      padding=padding_setting, vardefault=12, vartype=tk.DoubleVar,
                      infotext=it.baseline_smoothness, **common_kwargs)

    # baseline_polyorder
    baseline_polyorder_var, baseline_polyorder_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'Basline polyorder:',
                      padding=padding_setting, vardefault=2, vartype=tk.IntVar,
                      infotext=it.baseline_polyorder, **common_kwargs)

    # baseline_itermax
    baseline_itermax_var, baseline_itermax_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'Basline itermax:',
                      padding=padding_setting, vardefault=10, vartype=tk.IntVar,
                      infotext=it.baseline_itermax, **common_kwargs)

    # sgf_windowsize
    sgf_windowsize_var, sgf_windowsize_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitr, 'SGF Windowsize:',
                      padding=padding_setting, vardefault=101, vartype=tk.IntVar,
                      infotext=it.sgf_windowsize, **common_kwargs)

    # sgf_applications
    sgf_applications_var, sgf_applications_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitr, 'SGF Applications:',
                      padding=padding_setting, vardefault=2, vartype=tk.IntVar,
                      infotext=it.sgf_applications, **common_kwargs)

    # sgf_polyorder
    sgf_polyorder_var, sgf_polyorder_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitr, 'SGF polyorder:',
                      padding=padding_setting, vardefault=0, vartype=tk.IntVar,
                      infotext=it.sgf_polyorder, **common_kwargs)

    # hybrid_smoothing
    hybrid_smoothing_var, _, _, _, _, _ = labeled_options(frame_peak_fitr, 'Use hybrid smoothing:', padding=padding_setting,
                                                          side=tk.LEFT, vartype=tk.StringVar, vardefault=bool_options[1],
                                                          infotext=it.hybrid_smoothing, **common_kwargs)
    # headings
    heading_peak_finding = heading("Peak Finding", lvl=2, frame=frame_peak_fitl, padding=False)
    heading_noise_reduction = heading("Noise Reduction", lvl=2, frame=frame_peak_fitr, padding=False)

    # peak_height_min
    peak_height_min_var, peak_height_min_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'Peak height minimum: noise *',
                      padding=padding_setting, vardefault=1.2, vartype=tk.DoubleVar,
                      infotext=it.peak_height_min, **common_kwargs)

    # peak_prominence_min
    peak_prominence_min_var, peak_prominence_min_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'Peak prominence minimum: noise *',
                      padding=padding_setting, vardefault=1.2, vartype=tk.DoubleVar,
                      infotext=it.peak_prominence_min, **common_kwargs)

    # peak_ph_ratio_min
    peak_ph_ratio_min_var, peak_ph_ratio_min_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'Peak prominence-to-height minimum:',
                      padding=padding_setting, vardefault=0.5, vartype=tk.DoubleVar,
                      infotext=it.peak_ph_ratio_min, **common_kwargs)

    # recursive_noise_reduction
    recursive_noise_reduction_var, recursive_noise_reduction_frame, _, _, _, _ =\
        labeled_options(frame_peak_fitr, 'Recursively reduce noise:',
                        padding=padding_setting, vartype=tk.StringVar,
                        vardefault=bool_options[0],
                        infotext=it.recursive_noise_reduction, **common_kwargs)

    # max_noise_reduction_iter
    max_noise_reduction_iter_var, max_noise_reduction_iter_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitr, 'Max noise reduction iterations:',
                      padding=padding_setting, vardefault=10, vartype=tk.IntVar,
                      infotext=it.max_noise_reduction_iter, **common_kwargs)

    # regularization_ratio
    regularization_ratio_var, regularization_ratio_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitr, 'Noise reduction regularization factor:',
                      padding=padding_setting, vardefault=0.5, vartype=tk.DoubleVar,
                      infotext=it.regularization_ratio, **common_kwargs)

    heading_ml_settings = heading("ML Settings", lvl=2, frame=frame_peak_fitl, padding=False)
    ml_threshold_var, ml_threshold_frame, _, _, _, _ =\
        labeled_entry(frame_peak_fitl, 'ML confidence threshold:',
                      padding=padding_setting, vardefault=0.01, vartype=tk.DoubleVar,
                      infotext=it.ml_threshold, **common_kwargs)

    ml_weights_path_var, ml_weights_path_frame, _, _, _, _ = labeled_file_select(frame_peak_fit,
                                                                                 label='ML weights file:',
                                                                                 vardefault='default weights',
                                                                                 filetype='ml_weights',
                                                                                 infotext=it.ml_weights, side=tk.TOP, **common_kwargs)

    peak_fitting_objects_std = [heading_baseline,
                                baseline_smoothness_frame, baseline_polyorder_frame, baseline_itermax_frame,
                                heading_smoothing,
                                sgf_windowsize_frame, sgf_applications_frame, sgf_polyorder_frame,
                                heading_peak_finding,
                                peak_height_min_frame, peak_prominence_min_frame, peak_ph_ratio_min_frame,
                                heading_noise_reduction,
                                recursive_noise_reduction_frame, max_noise_reduction_iter_frame, regularization_ratio_frame]
    peak_fitting_objects_ml = [heading_ml_settings,
                               ml_threshold_frame, ml_weights_path_frame,
                               heading_smoothing,
                               sgf_windowsize_frame, sgf_applications_frame, sgf_polyorder_frame]
    for o in peak_fitting_objects_ml:
        if o not in peak_fitting_objects_std:
            o.pack_forget()

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
                                                   infotext=it.max_stretch, **common_kwargs)

    # num_stretches
    num_stretches_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretches per iteration: 10^',
                                                     padding=padding_setting, vardefault=4, vartype=tk.DoubleVar,
                                                     infotext=it.num_stretches, **common_kwargs)

    # nw_normalized
    nw_normalized_var, _, _, _, _, _ = labeled_options(frame_peak_matchr, 'Normalize frequency for peak matching:',
                                                       padding=padding_setting, vartype=tk.StringVar,
                                                       vardefault=bool_options[1],
                                                       command=update_peak_match_window_label,
                                                       infotext=it.nw_normalized, **common_kwargs)

    # peak_match_window
    # var, frame, label1, entry, label2, infolabel
    peak_match_window_var, _, _, _, peak_match_window_label2, peak_match_window_infobox =\
        labeled_entry(frame_peak_matchr,
                      'Max matching difference:',
                      postlabel='Hz',
                      padding=padding_setting, vardefault=150,
                      vartype=tk.DoubleVar,
                      infotext=it.peak_match_window, **common_kwargs)

    # matching_penalty_order
    matching_penalty_order_var, _, _, _, _, _ = labeled_entry(frame_peak_matchr,
                                                              'Matching penalty order:',
                                                              padding=padding_setting, vardefault=1, vartype=tk.DoubleVar,
                                                              infotext=it.matching_penalty_order, **common_kwargs)

    # dummy
    frame_dummy = tk.Frame(frame_peak_matchr)
    frame_dummy.pack(**padding_setting, side=tk.TOP)
    dummy_label = tk.Label(frame_dummy, text="")
    dummy_label.pack(side=tk.LEFT)

    frame_submit = tk.Frame(rootsubmit)
    frame_submit.pack(**padding_heading, side=tk.BOTTOM)

    submit_Button = tk.Button(frame_submit, text="Run Code", bg='firebrick4', fg='white',
                              width=20, height=2, command=submit, font=(default_font_name, 20, "bold"))
    submit_Button.pack(**padding_heading)

    frame_status = tk.Frame(frame_submit)
    frame_status.pack(side=tk.BOTTOM)

    status_label0 = tk.Label(frame_status, text='Status:', font=(default_font_name, 12))
    status_label0.pack(**padding_setting, side=tk.LEFT)
    status_label = tk.Label(frame_status, text='No directory selected.',
                            bg='firebrick4', fg='white', font=(default_font_name, 12))
    status_label.pack(**padding_setting, side=tk.RIGHT)

    make_button(rootbuttons, text="View Plots",
                command=lambda: open_plot_window(root, data_dict_var,
                                                 pair_results_var,
                                                 slc_limits_min_var,
                                                 slc_limits_max_var),
                padding=padding_heading)

    make_button(rootbuttons, text="Clustering",
                command=lambda: open_clustering_window(root, pair_results_var),
                padding=padding_heading)

    make_button(rootbuttons, text="Results Table",
                command=lambda: open_results_table_window(root, data_dict_var,
                                                          pair_results_var),
                padding=padding_heading)

    make_button(rootbuttons, text="Peak List",
                command=lambda: open_peak_list_window(root, data_dict_var, directory_var),
                padding=padding_heading)

    make_button(rootbuttons, text="Log",
                command=lambda: open_log_window(root, log_var, log_file_loc_var),
                padding=padding_heading)

    directory_var_prev = tk.StringVar(root, value='')

    log_file = tempfile.NamedTemporaryFile('w', delete=False)
    log_file.close()
    log_file_loc_var = tk.StringVar(root, value=log_file.name)

    def update_directory(*args):
        if directory_var.get() == directory_var_prev.get():
            return
        log_file_loc_var.set(osp.join(directory_var.get(), 'LARSAnalysisLog.log'))
        if using_temp_file.get():
            shmove(log_file.name,
                   log_file_loc_var.get())
        else:
            shmove(osp.join(directory_var_prev.get(), 'LARSAnalysisLog.log'),
                   log_file_loc_var.get())
        using_temp_file.set(False)
        directory_var_prev.set(directory_var.get())
        return

    directory_var.trace_add('write', update_directory)

    # Add printing to log
    log_var = tk.StringVar(root, value='')
    sys.stdout.write = log_decorator(sys.stdout.write, log_var, log_file_loc_var, running_var)

    print('opened app')

    canvas.update()
    canvas_frame = canvas.nametowidget(canvas.itemcget("frame_canvas", "window"))
    min_width = canvas_frame.winfo_reqwidth()
    min_height = canvas_frame.winfo_reqheight()
    canvas.itemconfigure("frame_canvas", width=min_width)
    canvas.itemconfigure("frame_canvas", height=min_height)
    canvas.configure(scrollregion=canvas.bbox("all"))

    # Start the main loop
    root.mainloop()


def run_app():
    try:
        run_app_main()
    except Exception as e:
        import traceback
        print('Error:', e)
        print(traceback.format_exc())
        raise (e)


if __name__ == '__main__':
    run_app()
