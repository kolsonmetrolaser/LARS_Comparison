# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:16:34 2024

@author: KOlson
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from plotfunctions import line_plot
    from app_helpers import CustomVar
except ModuleNotFoundError:
    from MetroLaserLARS.plotfunctions import line_plot
    from MetroLaserLARS.app_helpers import CustomVar


def open_plot_window(root, data_dict_var, pair_results_var, frange_min_var, frange_max_var, **common_kwargs):

    window = tk.Toplevel()
    # window.grab_set()
    window.title("Plots")

    f0, f1 = frange_min_var.get(), frange_max_var.get()
    common_kwargs['x_lim'] = (f0, f1)
    common_kwargs['v_line_width'] = 2
    common_kwargs['line_width'] = 4
    common_kwargs['show_plot_in_spyder'] = False
    common_kwargs['font_settings'] = {'weight': 'bold', 'size': 16}
    common_kwargs['x_slice'] = (1000, np.inf)

    data_dict, pair_results = data_dict_var.get(), pair_results_var.get()
    keys = list(data_dict.keys())
    if keys:
        k = keys[0] if keys else None
        data = data_dict[k] if k is not None else None
        x, y = data.freq, data.vel
        fig, [line] = line_plot(x, y, title='Raw Data', **common_kwargs)
    else:
        k = None
        data = None
        x, y = np.linspace(0, 1, 100), np.sin(2*np.pi*np.linspace(0, 1, 100))
        fig, [line] = line_plot(x, y, title='Example', show_plot_in_spyder=False)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()

    def my_key_press_handler(event):
        data2_options = data2_options_var.get()
        if event.key == 'right' or event._guiEvent.keycode == 97:
            plot_type_var.set(plot_type_options[min(plot_type_options.index(
                plot_type_var.get())+1, len(plot_type_options)-1)])
        elif event.key == 'left' or event._guiEvent.keycode == 100:
            plot_type_var.set(plot_type_options[max(plot_type_options.index(plot_type_var.get())-1, 0)])
        elif event.key == 'down' or event._guiEvent.keycode == 98:
            data_selection_var.set(data_options[min(data_options.index(
                data_selection_var.get())+1, len(data_options)-1)])
        elif event.key == 'up' or event._guiEvent.keycode == 101:
            data_selection_var.set(data_options[max(data_options.index(data_selection_var.get())-1, 0)])
        elif event._guiEvent.keycode == 99:
            data_selection2_var.set(data2_options[min(data2_options.index(
                data_selection2_var.get())+1, len(data2_options)-1)])
        elif event._guiEvent.keycode == 102:
            data_selection2_var.set(data2_options[max(data2_options.index(data_selection2_var.get())-1, 0)])
        else:
            key_press_handler(event)

    canvas.mpl_connect("key_press_event", my_key_press_handler)

    def update_data_selection2():
        data2_options = data_options.copy()
        data2_options.remove(data_selection_var.get())
        data2_options_var.set(data2_options)
        current_selection = data_selection2_var.get()
        data_selection2_menu.set_menu(current_selection if current_selection in data2_options else data2_options[0],
                                      *data2_options)

    def update_plot_contents(canvas, name_to_key, *args, **common_kwargs):
        update_data_selection2()

        kwargs = common_kwargs
        kwargs['fig'] = canvas.figure

        n = data_selection_var.get()
        n2 = data_selection2_var.get()
        t = plot_type_var.get()

        if n not in name_to_key:
            return
        if t in ['Compare Raw'] and n2 not in name_to_key:
            return

        # Get Data
        data = data_dict[name_to_key[n]]
        if t in ['Compare Raw']:
            data2 = data_dict[name_to_key[n2]]

        # Make Plots
        if t == 'Raw Data':
            x, y = data.freq, data.vel
            _ = line_plot(x, y, title='Raw Data', **kwargs)
        elif t == 'Peak Fits':
            x, y, p = data.freq, data.newvel, data.peaks['positions']
            f0, f1 = common_kwargs['x_lim']
            x = x[np.logical_and(x > f0, x < f1)]
            _ = line_plot(x, y, v_line_pos=p, title='Peak Fits', **kwargs)
        elif t == 'Compare Raw':
            x, y, n = data.freq, data.vel, data.name
            x2, y2, n2 = data2.freq, data2.vel, data2.name
            _ = line_plot([x, x2], [y, y2], legend=[n, n2], title='Peak Fits', **kwargs)

        # required to update canvas and attached toolbar!
        canvas.draw()

    frame_options = tk.Frame(window)

    plot_type_options = ['Raw Data', 'Peak Fits', 'Compare Raw']
    data_keys = list(data_dict.keys())
    if data_keys:
        name_to_key = {}
        data_options = []
        for k in data_keys:
            data_options.append(data_dict[k].name)
            name_to_key[data_dict[k].name] = k
        data2_options = data_options.copy()
        data2_options.pop(0)
    else:
        name_to_key = {'': ''}
        data_options = ['']
        data2_options = ['']
    data2_options_var = CustomVar()
    data2_options_var.set(data2_options)
    plot_type_var = tk.StringVar(window)
    plot_type_label = tk.Label(frame_options, text="Plot type:")
    plot_type_menu = ttk.OptionMenu(frame_options, plot_type_var, plot_type_options[0], *plot_type_options)

    data_selection_var = tk.StringVar(window)
    data_selection_label = tk.Label(frame_options, text="Data to Plot")
    data_selection_menu = ttk.OptionMenu(frame_options, data_selection_var, data_options[0], *data_options)

    data_selection2_var = tk.StringVar(window)
    data_selection2_label = tk.Label(frame_options, text="Data to Compare")
    data_selection2_menu = ttk.OptionMenu(frame_options, data_selection2_var,
                                          data2_options[0], *data2_options)

    plot_type_var.trace_add("write", lambda *args: update_plot_contents(canvas, name_to_key, **common_kwargs))
    data_selection_var.trace_add("write", lambda *args: update_plot_contents(canvas, name_to_key, **common_kwargs))
    data_selection2_var.trace_add("write", lambda *args: update_plot_contents(canvas, name_to_key, **common_kwargs))

    plot_type_label.pack(side=tk.LEFT)
    plot_type_menu.pack(side=tk.LEFT)
    data_selection_label.pack(side=tk.LEFT)
    data_selection_menu.pack(side=tk.LEFT)
    data_selection2_label.pack(side=tk.LEFT)
    data_selection2_menu.pack(side=tk.LEFT)

    frame_options.pack(side=tk.BOTTOM)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return
