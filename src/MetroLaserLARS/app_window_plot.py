# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:16:34 2024

@author: KOlson
"""
import tkinter as tk
import numpy as np
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
if True:  # __name__ == '__main__':
    from plotfunctions import line_plot
else:
    pass

if True:  # __name__ == '__main__':
    from app_helpers import labeled_options
else:
    pass


def open_plot_window(root, data_dict_var, pair_results_var, **common_kwargs):

    window = tk.Toplevel()
    window.grab_set()
    window.title("Plots")

    data_dict, pair_results = data_dict_var.get(), pair_results_var.get()

    t = np.arange(0, 3, .01)
    fig, [line] = line_plot(t, 2*np.sin(2*np.pi*t))

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()

    canvas.mpl_connect(
        "key_press_event", lambda event: print(f"you pressed {event.key}"))
    canvas.mpl_connect("key_press_event", key_press_handler)

    button_quit = tk.Button(master=window, text="Quit", command=window.destroy)

    def update_plot_contents(*args):
        # retrieve frequency
        f = float(data_selection_var.get())

        # update data
        y = 2 * np.sin(2 * np.pi * f * t)
        line.set_data(t, y)
        # fig = line_plot(t, y)

        # required to update canvas and attached toolbar!
        canvas.draw()

    frame_options = tk.Frame(window)

    options = [i+1 for i in range(5)]
    data_selection_var = tk.StringVar(window, value=options[0])
    data_selection_menu = tk.OptionMenu(frame_options, data_selection_var, *options, command=update_plot_contents)
    data_selection_menu.pack(side=tk.LEFT)

    button_quit.pack(side=tk.BOTTOM)
    frame_options.pack(side=tk.BOTTOM)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return
