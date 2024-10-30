# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# External imports
import tkinter as tk
from tkinter import filedialog, DoubleVar, StringVar, Label, Entry, Button, Text
from tkinter import OptionMenu, IntVar, Variable, BooleanVar, font, Toplevel

# Internal imports
if __name__ == '__main__':
    from infotext import infotext
    from LARS_Comparison import LARS_Comparison_from_app
else:
    from MetroLaserLARS.infotext import infotext
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
        x = x + self.widget.winfo_rootx() + 30
        y = y + cy + self.widget.winfo_rooty() + 30
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=10)
        tw.update()
        try:
            x, y, cx, cy = tw.winfo_rootx(), tw.winfo_rooty(), tw.winfo_width(), tw.winfo_height()
        except tk.TclError:
            return
        r = self.widget.winfo_toplevel()
        rx, ry, rcx, rcy = r.winfo_rootx(), r.winfo_rooty(), r.winfo_width(), r.winfo_height()
        # 20 px buffer from right and bottom edges
        if (rx+rcx) - (x + cx) < 20:
            x -= 20 - (rx+rcx) + (x + cx)
        if (ry+rcy) - (y + cy) < 20:
            y -= 20 - (ry+rcy) + (y + cy)
        tw.wm_geometry("+%d+%d" % (x, y))

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
            settings['directory']                 = directory_var.get() # noqa
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
        peak_match_window_infobox.pack_forget()
        txt = "Hz" if entry_text == 'False' else '*peak_position/Hz'
        peak_match_window_label2.config(text=txt)
        peak_match_window_infobox.pack(side=tk.LEFT)

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
            save_tag_infolabel.pack(side=tk.LEFT)
            # save_tag_dummy_label.pack(side=tk.BOTTOM)
        else:
            save_tag_label.pack_forget()
            save_tag_entry.pack_forget()
            save_tag_label2.pack_forget()
            save_tag_infolabel.pack_forget()

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
    bgc = 'gray65'
    root.config(bg=bgc)
    root.option_add("*Background", bgc)

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
        entry, infolabel = None, None
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
        return var, frame, label1, entry, label2, infolabel

    def labeled_options(baseframe=root, label: str = '', postlabel: str = '', padding=padding_setting,
                        var=None, vardefault=None, vartype=None, varframe=root, do_update_status=True,
                        command=None, side=tk.TOP, options=bool_options, infobox=True,
                        infotext='Placeholder info text.'):
        optionmenu, infolabel = None, None
        frame = labeled_widget_frame(baseframe, padding, side)
        label1 = labeled_widget_label(frame, label)
        if vartype is not None or var is not None:
            if var is not None and vardefault is not None:
                var.set(vardefault)
            if var is None:
                var = vartype(varframe, value=vardefault)
            if do_update_status:
                var.trace_add("write", update_status)
            optionmenu = OptionMenu(frame, var, *options, command=command)
            optionmenu.config(bg='gray75', highlightthickness=0)
            optionmenu.pack(side=tk.LEFT)
        label2 = labeled_widget_label(frame, postlabel)
        if infobox:
            infolabel = labeled_widget_label(frame, '(?)')
            CreateToolTip(infolabel, infotext)
        return var, frame, label1, optionmenu, label2, infolabel

    def labeled_file_select(baseframe=root, headingtxt: str = '', subheading: str = '', label: str = '',
                            padding=padding_setting, entry_width: int = 40, vardefault='', vartype=StringVar,
                            varframe=root, do_update_status=True, command=None, side=tk.TOP, selection='file',
                            filetype='pickle', infobox=True, infotext='Placeholder info text.'):
        entry, button, infolabel = None, None, None
        frame = labeled_widget_frame(baseframe, padding, side)
        if headingtxt != '':
            heading(headingtxt, lvl=1, frame=frame,
                    subtext=subheading)
        elif subheading != '':
            heading(subheading, lvl=3, frame=frame)
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
        return var, frame, label1, entry, button, infolabel

    def part_matching_window(root, grouped_folders_var, part_matching_text_var):
        def toggle_code_labels(*args):
            if part_matching_strategy_var.get() == 'custom':
                try:
                    code_label_1.config(text="""Define a custom Python function given the names of the folders which contain .all files.
name0 and name1 are strings and should be treated symmetrically; there is no guarantee
a particular folder will be associated with one. The function must define the variable
result, which must evaluate to True when the parts match and False when they do not.

For example, if parts match if and only if the first letter of their folder name matches,
enter the following below:

result = name0[0] == name1[0]

def part_matching_function(name0, name1):""", font=("Courier New", 9))
                    code_label_1.pack_forget()
                    code_label_2.pack_forget()
                    code_label_3.pack_forget()
                    part_matching_text.pack_forget()
                    frame_text.pack_forget()

                    code_label_1.pack(anchor='w')
                    frame_text.pack()
                    code_label_2.pack(side=tk.LEFT)
                    part_matching_text.pack(side=tk.LEFT)
                    code_label_3.pack(anchor='w')
                except tk.TclError:
                    pass
            elif part_matching_strategy_var.get() == 'list':
                try:
                    code_label_1.config(text="""Enter matching parts separated by a comma and space, and each part group on a new line.
The lists should contain only the names of the folders which contain .all files.

For example:
    
Group1PartA, Group1PartB, Group1PartC
Group2PartA, Group2PartB""", font=("Courier New", 9), justify='left')
                    code_label_1.pack_forget()
                    code_label_2.pack_forget()
                    code_label_3.pack_forget()
                    part_matching_text.pack_forget()
                    frame_text.pack_forget()

                    code_label_1.pack(anchor='w')
                    frame_text.pack()
                    part_matching_text.pack(side=tk.LEFT)
                except tk.TclError:
                    pass
            else:
                try:
                    code_label_1.pack_forget()
                    code_label_2.pack_forget()
                    code_label_3.pack_forget()
                    part_matching_text.pack_forget()
                    frame_text.pack_forget()

                    frame_text.pack()
                    part_matching_text.pack(side=tk.LEFT)
                except tk.TclError:
                    pass
            return

        window = Toplevel()
        window.grab_set()
        window.title("Define Known Part Matching")
        label1 = Label(window,
                       text="""
Define known part matching by entering a list of equivalent parts,
by creating a custom function, or by the folder structure (if "Use
grouped folder structure" is set to True).""")
        label1.pack()
        options = ['folder', 'list', 'custom'] if grouped_folders_var.get() == 'True' else ['list', 'custom']
        _, _, _, _, _, _ = labeled_options(window, 'Definition type:',
                                           var=part_matching_strategy_var,
                                           options=options,
                                           infobox=False)

        frame_input = tk.Frame(window)
        frame_input.pack()

        code_label_1 = Label(frame_input, text='def part_matching_function(name0, name1):', font=("Courier New", 9))
        code_label_1.pack(anchor="w")

        frame_text = tk.Frame(frame_input)
        frame_text.pack()

        code_label_2 = Label(frame_text, text='    ', font=("Courier New", 9))
        code_label_2.pack(side=tk.LEFT)

        part_matching_text = Text(frame_text)
        part_matching_text.insert("0.0", part_matching_text_var.get())
        part_matching_text.pack(side=tk.LEFT)

        code_label_3 = Label(frame_input, text='return result', font=("Courier New", 9))
        code_label_3.pack(anchor="w")

        part_matching_text_submit = Button(window, text='Save entered text', command=lambda: part_matching_text_var.set(part_matching_text.get("0.0", tk.END)))
        part_matching_text_submit.pack()

        fn = toggle_code_labels
        part_matching_strategy_var.trace_add('write', fn)
        toggle_code_labels()

        return part_matching_text_var.get()

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
                                                       infotext=infotext['directory'])
    # pickled_data_path

    pickled_data_path_var, _, _, _, _, _ = labeled_file_select(rootload, headingtxt='Pickled Data',
                                                               subheading='Load data from previous analysis.',
                                                               label='Enter path to pickled data or select a file:',
                                                               infotext=infotext['pickled_data_path'])
    heading('Settings', frame=roottop, side=tk.BOTTOM)

    # DATA DEFINITIONS

    heading("Data", lvl=1, frame=rootl, side=tk.TOP)

    # frange and slc limits
    frame_frange = tk.Frame(rootl)
    frame_frange.pack(**padding_setting, side=tk.TOP)
    frange_min_var, _, _, _, _, _ = labeled_entry(frame_frange, 'Data minimum and maximum frequency:',
                                                  padding=padding_none,
                                                  vardefault=10000, vartype=DoubleVar, side=tk.LEFT,
                                                  infobox=False)
    frange_max_var, _, _, _, _, _ = labeled_entry(frame_frange, '-',
                                                  postlabel='Hz', padding=padding_none,
                                                  vardefault=60000, vartype=DoubleVar, side=tk.LEFT,
                                                  infotext=infotext['frange'])

    # combine
    combine_var, _, _, _, _, _ = labeled_options(rootl, 'How data within a folder should be combined:',
                                                 padding=padding_setting, vartype=StringVar,
                                                 vardefault='max', options=['max', 'mean'],
                                                 infotext=infotext['combine'])

    # grouped_folders
    grouped_folders_var, _, _, _, _, _ = labeled_options(rootr, 'Use grouped folder structure:', padding=padding_setting,
                                                         vartype=StringVar, vardefault=bool_options[1],
                                                         infotext=infotext['grouped_folders'])

    # part_matching
    frame_part_matching = tk.Frame(rootr)
    frame_part_matching.pack(**padding_setting)

    part_matching_text_var = StringVar(root, value='')
    part_matching_strategy_var = StringVar(root, value='list')

    part_matching_button = Button(rootr, text="Define Known Part Matching", bg='gray75',
                                  command=lambda: part_matching_text_var.set(part_matching_window(root, grouped_folders_var, part_matching_text_var)))
    part_matching_button.pack()

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
    plot_detail_var = StringVar(root, value=bool_options[1])
    plot_recursive_noise_var = StringVar(root, value=bool_options[1])
    plot_classification_var = StringVar(root, value=bool_options[1])

    plot_var.trace_add("write", update_status)
    plot_detail_var.trace_add("write", update_status)
    plot_recursive_noise_var.trace_add("write", update_status)
    plot_classification_var.trace_add("write", update_status)

    plot_menu = OptionMenu(frame_plot_menus, plot_var, *bool_options)
    plot_detail_menu = OptionMenu(frame_plot_menus, plot_detail_var, *bool_options)
    plot_recursive_noise_menu = OptionMenu(frame_plot_menus, plot_recursive_noise_var, *bool_options)
    plot_classification_menu = OptionMenu(frame_plot_menus, plot_classification_var, *bool_options)

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
                                                       vardefault=20, vartype=DoubleVar,
                                                       infotext=infotext['peak_plot_width'])

    # show_plots
    frame_show_save_plots = tk.Frame(rootl)
    frame_show_save_plots.pack(side=tk.TOP)
    show_plots_var, _, _, _, _, _ = labeled_options(frame_show_save_plots, 'Show plots:', padding=padding_setting,
                                                    side=tk.LEFT, vartype=StringVar, vardefault=bool_options[1],
                                                    infotext=infotext['show_plots'])

    # save_plots
    save_plots_var, _, _, _, _, _ = labeled_options(frame_show_save_plots, 'Save plots:', padding=padding_setting,
                                                    side=tk.LEFT, vartype=StringVar, vardefault=bool_options[1],
                                                    infotext=infotext['save_plots'])

    # PRINT_MODE
    PRINT_MODE_var, _, _, _, _, _ = labeled_options(rootl, 'Print details:',
                                                    padding=padding_setting, vartype=StringVar,
                                                    vardefault='sparse', options=['none', 'sparse', 'full'],
                                                    infotext=infotext['PRINT_MODE'])
    # SAVING

    heading("Saving", lvl=1, frame=rootl, padding=False)

    # save_data
    save_data_var, _, _, _, _, _ = labeled_options(rootl, 'Save data to .pkl file:', padding=padding_setting,
                                                   vartype=StringVar, vardefault=bool_options[1], command=hide_save_tag,
                                                   infotext=infotext['save_data'])

    # save_results
    save_results_var, _, _, _, _, _ = labeled_options(rootl, 'Save results to .pkl file:', padding=padding_setting,
                                                      vartype=StringVar, vardefault=bool_options[0], command=hide_save_tag,
                                                      infotext=infotext['save_results'])

    # save_tag
    save_tag_var, frame_save_tag, save_tag_label, save_tag_entry, save_tag_label2, save_tag_infolabel =\
        labeled_entry(rootl, 'Save filename: data_dict... and peak_results', postlabel='.pkl', padding=padding_setting,
                      vardefault='', vartype=StringVar,
                      infotext=infotext['save_tag'])
    save_tag_var.trace_add("write", update_save_tag_label)

    save_tag_dummy_label = Label(frame_save_tag, text=" "*42, font=("Courier New", 9))

    # save_directory
    save_directory_var, _, _, _, _, _ = labeled_file_select(rootl, subheading='Enter path or select a folder to save results, data, and/or plots:',
                                                            selection='dir', vardefault='Same as LARS Data Directory',
                                                            infotext=infotext['save_folder'])

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
                                                           padding=padding_setting, vardefault=12, vartype=DoubleVar,
                                                           infotext=infotext['baseline_smoothness'])

    # baseline_polyorder
    baseline_polyorder_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline polyorder:',
                                                          padding=padding_setting, vardefault=2, vartype=IntVar,
                                                          infotext=infotext['baseline_polyorder'])

    # baseline_itermax
    baseline_itermax_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Basline itermax:',
                                                        padding=padding_setting, vardefault=10, vartype=IntVar,
                                                        infotext=infotext['baseline_itermax'])

    # sgf_windowsize
    sgf_windowsize_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF Windowsize:',
                                                      padding=padding_setting, vardefault=101, vartype=IntVar,
                                                      infotext=infotext['sgf_windowsize'])

    # sgf_applications
    sgf_applications_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF Applications:',
                                                        padding=padding_setting, vardefault=2, vartype=IntVar,
                                                        infotext=infotext['sgf_applications'])

    # sgf_polyorder
    sgf_polyorder_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'SGF polyorder:',
                                                     padding=padding_setting, vardefault=0, vartype=IntVar,
                                                     infotext=infotext['sgf_polyorder'])

    # headings
    heading("Peak Finding", lvl=2, frame=frame_peak_fitl, padding=False)
    heading("Noise Reduction", lvl=2, frame=frame_peak_fitr, padding=False)

    # peak_height_min
    peak_height_min_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak height minimum: noise *:',
                                                       padding=padding_setting, vardefault=0.2, vartype=DoubleVar,
                                                       infotext=infotext['peak_height_min'])

    # peak_prominence_min
    peak_prominence_min_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak prominence minimum: noise *:',
                                                           padding=padding_setting, vardefault=0.2, vartype=DoubleVar,
                                                           infotext=infotext['peak_prominence_min'])

    # peak_ph_ratio_min
    peak_ph_ratio_min_var, _, _, _, _, _ = labeled_entry(frame_peak_fitl, 'Peak prominence-to-height minimum:',
                                                         padding=padding_setting, vardefault=0.5, vartype=DoubleVar,
                                                         infotext=infotext['peak_ph_ratio_min'])

    # recursive_noise_reduction
    recursive_noise_reduction_var, _, _, _, _, _ = labeled_options(frame_peak_fitr, 'Recursively reduce noise:',
                                                                   padding=padding_setting, vartype=StringVar,
                                                                   vardefault=bool_options[0],
                                                                   infotext=infotext['recursive_noise_reduction'])

    # max_noise_reduction_iter
    max_noise_reduction_iter_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'Max noise reduction iterations:',
                                                                padding=padding_setting, vardefault=10, vartype=IntVar,
                                                                infotext=infotext['max_noise_reduction_iter'])

    # regularization_ratio
    regularization_ratio_var, _, _, _, _, _ = labeled_entry(frame_peak_fitr, 'Noise reduction regularization factor:',
                                                            padding=padding_setting, vardefault=0.5, vartype=DoubleVar,
                                                            infotext=infotext['regularization_ratio'])

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
                                                   padding=padding_setting, vardefault=0.02, vartype=DoubleVar,
                                                   infotext=infotext['max_stretch'])

    # num_stretches
    num_stretches_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretches per iteration:',
                                                     padding=padding_setting, vardefault=1000, vartype=IntVar,
                                                     infotext=infotext['num_stretches'])

    # stretching_iterations
    stretching_iterations_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl, 'Number of stretching iterations:',
                                                             padding=padding_setting, vardefault=10, vartype=IntVar,
                                                             infotext=infotext['stretching_iterations'])

    # stretch_iteration_factor
    stretch_iteration_factor_var, _, _, _, _, _ = labeled_entry(frame_peak_matchl,
                                                                'Factor to reduce stretch space each iteration:',
                                                                padding=padding_setting, vardefault=5,
                                                                vartype=DoubleVar,
                                                                infotext=infotext['stretch_iteration_factor'])

    # nw_normalized
    nw_normalized_var, _, _, _, _, _ = labeled_options(frame_peak_matchr, 'Normalize frequency for peak matching:',
                                                       padding=padding_setting, vartype=StringVar,
                                                       vardefault=bool_options[1],
                                                       command=update_peak_match_window_label,
                                                       infotext=infotext['nw_normalized'])

    # peak_match_window
    # var, frame, label1, entry, label2, infolabel
    peak_match_window_var, _, _, _, peak_match_window_label2, peak_match_window_infobox =\
        labeled_entry(frame_peak_matchr,
                      'Max matching difference:',
                      postlabel='Hz',
                      padding=padding_setting, vardefault=150,
                      vartype=DoubleVar,
                      infotext=infotext['peak_match_window'])

    # matching_penalty_order
    matching_penalty_order_var, _, _, _, _, _ = labeled_entry(frame_peak_matchr,
                                                              'Matching penalty order:',
                                                              padding=padding_setting, vardefault=1, vartype=DoubleVar,
                                                              infotext=infotext['matching_penalty_order'])

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
    run_app()
