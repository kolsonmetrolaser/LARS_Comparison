# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:16:34 2024

@author: KOlson
"""
import tkinter as tk
import numpy as np
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from plotfunctions import line_plot
except ModuleNotFoundError:
    from MetroLaserLARS.plotfunctions import line_plot


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
    k = keys[0] if keys else None
    data = data_dict[k] if k is not None else None
    x, y = (data.freq, data.vel) if data is not None else (np.linspace(0, 1, 100), np.sin(2*np.pi*np.linspace(0, 1, 100)))
    fig, [line] = line_plot(x, y, title='Raw Data' if data is not None else 'Example', **common_kwargs)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()

    def my_key_press_handler(event):
        # print(f"you pressed {event.key}")
        if event.key == 'right':
            plot_type_var.set(plot_type_options[min(plot_type_options.index(plot_type_var.get())+1, len(plot_type_options)-1)])
        elif event.key == 'left':
            plot_type_var.set(plot_type_options[max(plot_type_options.index(plot_type_var.get())-1, 0)])
        elif event.key == 'down':
            data_selection_var.set(data_options[min(data_options.index(data_selection_var.get())+1, len(data_options)-1)])
        elif event.key == 'up':
            data_selection_var.set(data_options[max(data_options.index(data_selection_var.get())-1, 0)])
        else:
            key_press_handler(event)

    canvas.mpl_connect("key_press_event", my_key_press_handler)

    def update_plot_contents(canvas, name_to_key, *args, **common_kwargs):
        n = data_selection_var.get()
        if n not in name_to_key:
            return
        data = data_dict[name_to_key[n]]
        t = plot_type_var.get()
        if t == 'Raw Data':
            x, y = data.freq, data.vel
            _ = line_plot(x, y, fig=canvas.figure, title='Raw Data', **common_kwargs)
        elif t == 'Peak Fits':
            x, y, p = data.freq, data.newvel, data.peaks['positions']
            f0, f1 = common_kwargs['x_lim']
            x = x[np.logical_and(x > f0, x < f1)]
            _ = line_plot(x, y, v_line_pos=p, fig=canvas.figure, title='Peak Fits', **common_kwargs)

        # required to update canvas and attached toolbar!
        canvas.draw()

    frame_options = tk.Frame(window)

    plot_type_options = ['Raw Data', 'Peak Fits']
    data_keys = list(data_dict.keys())
    if data_keys:
        name_to_key = {}
        data_options = []
        for k in data_keys:
            data_options.append(data_dict[k].name)
            name_to_key[data_dict[k].name] = k
    else:
        name_to_key = {'': ''}
        data_options = ['']
    plot_type_var = tk.StringVar(window, value=plot_type_options[0])
    data_selection_var = tk.StringVar(window, value=data_options[0])
    plot_type_label = tk.Label(frame_options, text="Plot type:")
    plot_type_menu = tk.OptionMenu(frame_options, plot_type_var, *plot_type_options)
    data_selection_label = tk.Label(frame_options, text="Data to Plot")
    data_selection_menu = tk.OptionMenu(frame_options, data_selection_var, *data_options)

    plot_type_var.trace_add("write", lambda *args: update_plot_contents(canvas, name_to_key, **common_kwargs))
    data_selection_var.trace_add("write", lambda *args: update_plot_contents(canvas, name_to_key, **common_kwargs))

    plot_type_label.pack(side=tk.LEFT)
    plot_type_menu.pack(side=tk.LEFT)
    data_selection_label.pack(side=tk.LEFT)
    data_selection_menu.pack(side=tk.LEFT)

    frame_options.pack(side=tk.BOTTOM)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return
