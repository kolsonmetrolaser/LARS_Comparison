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


class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


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

    # def select_directory(entry, **kwargs):
    def select_directory(entry):
        directory = filedialog.askdirectory(title="Select a Directory")
        if directory:
            entry.delete(0, tk.END)  # Clear the directory_entry box
            entry.insert(0, directory)  # Insert the selected directory

    # def select_save_directory():
    #     save_directory = filedialog.askdirectory(title="Select a Directory")
    #     if save_directory:
    #         save_directory_entry.delete(0, tk.END)  # Clear the save_directory_entry box
    #         save_directory_entry.insert(0, save_directory)  # Insert the selected save_directory

    def select_file(entry, **kwargs):
        path = filedialog.askopenfilename(**kwargs)
        if path:
            entry.delete(0, tk.END)  # Clear the pickled_data_path_entry box
            entry.insert(0, path)  # Insert the selected pickled_data_path

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
    root.state('zoomed')

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

    # Global info
    bool_options = ['True', 'False']
    padding_heading = {'pady': 10, 'padx': 10}
    padding_setting = {'pady': 4, 'padx': 4}
    padding_option = {'pady': 0, 'padx': 4}
    padding_none = {'pady': 0, 'padx': 0}
    running_var = BooleanVar(root, value=False)
    prev_settings_var = Variable(root, value={})
    status_var = StringVar(root, value='nodir')
    default_font_name = font.nametofont('TkTextFont').actual()['family']
    kwargs_pickle = {'title': "Select a data_dict[...].pkl file",
                     'filetypes': [("Pickled Data Dictionaries", "data_dict*.pkl"), ("All Files", "*.*")]}

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
            h.pack(**padding_heading, side=side, fill='x')
        else:
            h.pack(side=side, fill='x')
        return h

    def labeled_widget_frame(baseframe, padding, side):
        lwframe = tk.Frame(baseframe)
        lwframe.pack(**padding, side=side)
        return lwframe

    def labeled_widget_label(frame, text):
        label = None
        if text != '':
            label = Label(frame, text=text)
            label.pack(side=tk.LEFT)
            if text == '(?)':
                f = font.Font(label, label.cget("font"))
                f.configure(underline=True)
                label.configure(font=f)
        return label

    def labeled_entry(baseframe=root, label: str = '', postlabel: str = '', padding=padding_setting,
                      entry_width: int = 6, vardefault=0, vartype=None, varframe=root, do_update_status=True,
                      side=tk.TOP, infobox=True, infotext='Placeholder info text.'):
        entry = None
        frame = labeled_widget_frame(baseframe, padding, side)
        label1 = labeled_widget_label(frame, label)
        if vartype is not None:
            var = vartype(varframe, value=vardefault)
            if do_update_status:
                var.trace_add("write", update_status)
            entry = Entry(frame, width=entry_width, textvariable=var)
            entry.pack(side=tk.LEFT)
        label2 = labeled_widget_label(frame, postlabel)
        if infobox:
            infolabel = labeled_widget_label(frame, '(?)')
            CreateToolTip(infolabel, infotext)
        return var, frame, label1, entry, label2

    def labeled_options(baseframe=root, label: str = '', postlabel: str = '', padding=padding_setting,
                        vardefault=0, vartype=None, varframe=root, do_update_status=True,
                        command=None, side=tk.TOP, options=bool_options, infobox=True,
                        infotext='Placeholder info text.'):
        optionmenu = None
        frame = labeled_widget_frame(baseframe, padding, side)
        label1 = labeled_widget_label(frame, label)
        if vartype is not None:
            var = vartype(varframe, value=vardefault)
            if do_update_status:
                var.trace_add("write", update_status)
            optionmenu = OptionMenu(frame, var, *options, command=command)
            optionmenu.config(bg='gray75')
            optionmenu.pack(side=tk.LEFT)
        label2 = labeled_widget_label(frame, postlabel)
        if infobox:
            infolabel = labeled_widget_label(frame, '(?)')
            CreateToolTip(infolabel, infotext)
        return var, frame, label1, optionmenu, label2

    def labeled_file_select(baseframe=root, headingtxt: str = '', subheading: str = '', label: str = '',
                            padding=padding_setting, entry_width: int = 40, vardefault='', vartype=StringVar,
                            varframe=root, do_update_status=True, command=None, side=tk.TOP, selection='file',
                            filetype='pickle', infobox=True, infotext='Placeholder info text.'):
        entry, button = None, None
        frame = labeled_widget_frame(baseframe, padding, side)
        if headingtxt != '':
            heading(headingtxt, lvl=1, frame=frame,
                    subtext=subheading)
        label1 = labeled_widget_label(frame, label)
        if vartype is not None:
            var = vartype(varframe, value=vardefault)
            if do_update_status:
                var.trace_add("write", update_status)
            entry = Entry(frame, width=entry_width, textvariable=var)
            if selection == 'file':
                fun = select_file
                if filetype == 'pickle':
                    kwargs = kwargs_pickle
            elif selection == 'dir':
                kwargs = {}
                fun = select_directory
            button = Button(frame, text="Open", command=lambda: fun(entry, **kwargs), bg='gray75')
            button.pack(side=tk.LEFT, padx=4)
            entry.pack(side=tk.LEFT, padx=4)
            if infobox:
                infolabel = labeled_widget_label(frame, '(?)')
                CreateToolTip(infolabel, infotext)
        return var, frame, label1, entry, button

    # Start building App

    heading('Load Data', frame=roottop, padding=False, side=tk.TOP)
    rootload = tk.Frame(roottop)
    rootload.pack(side=tk.TOP)
    # DIRECTORY
    directory_var, _, _, _, _ = labeled_file_select(rootload, headingtxt='Directory',
                                                    subheading="""Select a folder which contains subfolders, each of which contain LARS data in .all format.
All pairs of subfolders will be compared.""",
                                                    label='Enter path to LARS data or select a folder:', selection='dir')
    # pickled_data_path

    pickled_data_path_var, _, _, _, _ = labeled_file_select(rootload, headingtxt='Pickled Data',
                                                            subheading='Load data from previous analysis.',
                                                            label='Enter path to pickled data or select a file:')
    heading('Settings', frame=roottop, side=tk.BOTTOM)

    # DATA DEFINITIONS

    heading("Data", lvl=1, frame=rootl, side=tk.TOP)

    # frange and slc limits
    frame_frange = tk.Frame(rootl)
    frame_frange.pack(**padding_setting, side=tk.TOP)
    frange_min_var, _, _, _, _ = labeled_entry(frame_frange, 'Data minimum and maximum frequency:',
                                               padding=padding_none,
                                               vardefault=10000, vartype=DoubleVar, side=tk.LEFT)
    frange_max_var, _, _, _, _ = labeled_entry(frame_frange, '-',
                                               postlabel='Hz', padding=padding_none,
                                               vardefault=60000, vartype=DoubleVar, side=tk.LEFT)

    # combine
    combine_var, _, _, _, _ = labeled_options(rootl, 'How data within a folder should be combined:',
                                              padding=padding_setting, vartype=StringVar,
                                              vardefault='max', options=['max', 'mean'])

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
    peak_plot_width_var, _, _, _, _ = labeled_entry(rootl, 'Width of peak fit plots:',
                                                    postlabel='kHz', padding=padding_setting,
                                                    vardefault=20, vartype=DoubleVar)

    # show_plots
    frame_show_save_plots = tk.Frame(rootl)
    frame_show_save_plots.pack(side=tk.TOP)
    show_plots_var, _, _, _, _ = labeled_options(frame_show_save_plots, 'Show plots:', padding=padding_setting,
                                                 side=tk.LEFT, vartype=StringVar, vardefault=bool_options[1])

    # save_plots
    save_plots_var, _, _, _, _ = labeled_options(frame_show_save_plots, 'Save plots:', padding=padding_setting,
                                                 side=tk.LEFT, vartype=StringVar, vardefault=bool_options[1])

    # PRINT_MODE
    PRINT_MODE_var, _, _, _, _ = labeled_options(rootl, 'Print details:',
                                                 padding=padding_setting, vartype=StringVar,
                                                 vardefault='sparse', options=['none', 'sparse', 'full'])
    # SAVING

    heading("Saving", lvl=1, frame=rootl, padding=False)

    # save_data
    save_data_var, _, _, _, _ = labeled_options(rootl, 'Save data to .pkl file:', padding=padding_setting,
                                                vartype=StringVar, vardefault=bool_options[1], command=hide_save_tag)

    # save_results
    save_results_var, _, _, _, _ = labeled_options(rootl, 'Save results to .pkl file:', padding=padding_setting,
                                                   vartype=StringVar, vardefault=bool_options[0], command=hide_save_tag)

    # save_tag
    save_tag_var, frame_save_tag, save_tag_label, save_tag_entry, save_tag_label2 =\
        labeled_entry(rootl, 'Save filename: data_dict... and peak_results', postlabel='.pkl', padding=padding_setting,
                      vardefault='', vartype=StringVar)

    save_tag_dummy_label = Label(frame_save_tag, text=" "*42, font=("Courier New", 9))

    # save_directory
    save_directory_var, _, _, _, _ = labeled_file_select(rootl, subheading='Enter path to save data to or select a folder:',
                                                         selection='dir', vardefault='Same as LARS Data Directory')

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
    baseline_smoothness_var, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline smoothness: 10^',
                                                        padding=padding_setting, vardefault=12, vartype=DoubleVar)

    # baseline_polyorder
    baseline_polyorder_var, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline polyorder:',
                                                       padding=padding_setting, vardefault=2, vartype=IntVar)

    # baseline_itermax
    baseline_itermax_var, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline itermax:',
                                                     padding=padding_setting, vardefault=10, vartype=IntVar)

    # sgf_windowsize
    sgf_windowsize_var, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF Windowsize:',
                                                   padding=padding_setting, vardefault=101, vartype=IntVar)

    # sgf_applications
    sgf_applications_var, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF Applications:',
                                                     padding=padding_setting, vardefault=2, vartype=IntVar)

    # sgf_polyorder
    sgf_polyorder_var, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF polyorder:',
                                                  padding=padding_setting, vardefault=0, vartype=IntVar)

    # headings
    heading("Peak Finding", lvl=2, frame=frame_peak_fitl, padding=False)
    heading("Noise Reduction", lvl=2, frame=frame_peak_fitr, padding=False)

    # peak_height_min
    peak_height_min_var, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak height minimum: noise *:',
                                                    padding=padding_setting, vardefault=0.2, vartype=DoubleVar)

    # peak_prominence_min
    peak_prominence_min_var, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak prominence minimum: noise *:',
                                                        padding=padding_setting, vardefault=0.2, vartype=DoubleVar)

    # peak_ph_ratio_min
    peak_ph_ratio_min_var, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak prominence-to-height minimum:',
                                                      padding=padding_setting, vardefault=0.5, vartype=DoubleVar)

    # recursive_noise_reduction
    recursive_noise_reduction_var, _, _, _, _ = labeled_options(frame_peak_fitr, 'Recursively reduce noise:',
                                                                padding=padding_setting, vartype=StringVar,
                                                                vardefault=bool_options[0])

    # max_noise_reduction_iter
    max_noise_reduction_iter_var, _, _, _, _ = labeled_entry(frame_peak_fitr, 'Max noise reduction iterations:',
                                                             padding=padding_setting, vardefault=10, vartype=IntVar)

    # regularization_ratio
    regularization_ratio_var, _, _, _, _ = labeled_entry(frame_peak_fitr, 'Noise reduction regularization factor:',
                                                         padding=padding_setting, vardefault=0.5, vartype=DoubleVar)

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
    max_stretch_var, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Max stretching: 1 ±',
                                                padding=padding_setting, vardefault=0.02, vartype=DoubleVar)

    # num_stretches
    num_stretches_var, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretches per iteration:',
                                                  padding=padding_setting, vardefault=1000, vartype=IntVar)

    # stretching_iterations
    stretching_iterations_var, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretching iterations:',
                                                          padding=padding_setting, vardefault=10, vartype=IntVar)

    # stretch_iteration_factor
    stretch_iteration_factor_var, _, _, _, _ = labeled_entry(frame_peak_matchl,
                                                             'Factor to reduce stretch space each iteration:',
                                                             padding=padding_setting, vardefault=5,
                                                             vartype=DoubleVar)

    # nw_normalized
    nw_normalized_var, _, _, _, _ = labeled_options(frame_peak_matchr, 'Normalize distance for peak matching:',
                                                    padding=padding_setting, vartype=StringVar,
                                                    vardefault=bool_options[1],
                                                    command=update_peak_match_window_label)

    # peak_match_window
    peak_match_window_var, _, _, _, peak_match_window_label2 = labeled_entry(frame_peak_matchr,
                                                                             'Factor to reduce stretch space each iteration:',
                                                                             postlabel='Hz',
                                                                             padding=padding_setting, vardefault=150,
                                                                             vartype=DoubleVar)

    # matching_penalty_order
    matching_penalty_order_var, _, _, _, _ = labeled_entry(frame_peak_matchr,
                                                           'Matching penalty order:',
                                                           padding=padding_setting, vardefault=1, vartype=DoubleVar)

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
