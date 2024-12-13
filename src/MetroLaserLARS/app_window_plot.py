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
    from plotfunctions import line_plot, make_legend_interactive
    from app_helpers import CustomVar, padding_setting, plot_style_widget, icon_ML, edit_name_menu_bar
except ModuleNotFoundError:
    from MetroLaserLARS.plotfunctions import line_plot, make_legend_interactive  # type: ignore
    from MetroLaserLARS.app_helpers import CustomVar, padding_setting, plot_style_widget, icon_ML, edit_name_menu_bar  # type: ignore


def open_plot_window(root, data_dict_var, pair_results_var, frange_min_var, frange_max_var, **common_kwargs):

    window = tk.Toplevel(root)
    window.title("Plots")
    window.geometry("1600x900")
    window.wm_iconphoto(False, tk.PhotoImage(file=icon_ML))
    edit_name_menu_bar(window)

    color_vars, style_vars = CustomVar(), CustomVar()

    f0, f1 = frange_min_var.get(), frange_max_var.get()
    common_kwargs['x_lim'] = (f0/1000, f1/1000)
    common_kwargs['line_width'] = 4
    common_kwargs['show_plot_in_spyder'] = False
    common_kwargs['font_settings'] = {'weight': 'bold', 'size': 16}
    common_kwargs['x_slice'] = (1e-10, np.inf)
    common_kwargs['legend_interactive'] = True
    common_kwargs['autoconvert_to_vlines'] = True

    data_dict, pair_results = data_dict_var.get(), pair_results_var.get()
    keys = list(data_dict.keys())
    if keys:
        k = keys[0]
        data = data_dict[k]
        x, y = data.freq, data.vel
        fig = line_plot(x/1000, y, legend=[data.name], x_label='Frequency (kHz)', y_label='Intensity (µm/s)',
                        title='Raw Data', **common_kwargs)
    else:
        k = None
        data = None
        x, y = np.linspace(0, 1, 100), np.sin(2*np.pi*np.linspace(0, 1, 100))
        fig = line_plot(x, y, legend=['data'], legend_interactive=True, title='Example', show_plot_in_spyder=False)

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
        data_selection2_menu.set_menu(current_selection
                                      if current_selection in data2_options
                                      else (data2_options[0] if data2_options else ''),
                                      *data2_options)

    def update_plot_style(*args, **kwargs):
        lines = ([line for axis in canvas.figure.axes for line in axis.get_lines()]
                 + [line for legend in canvas.figure.legends for line in legend.get_lines()])
        line_names = [line.get_label() for legend in canvas.figure.legends for line in legend.get_lines()]
        for line_name in line_names:
            color = color_vars.get()[line_name].get()
            style = style_vars.get()[line_name].get()
            for line in lines:
                if line.get_label() == line_name or (line.get_label()[0] == '_' and line.get_label()[1:] == line_name):
                    line.set_color(color)
                    if style in ['-', ':', '--', '-.']:
                        line.set_linestyle(style)
                        line.set_marker('None')
                    else:
                        line.set_marker(style)
                        line.set_linestyle('None')
            canvas.draw_idle()
        return

    def update_style_options():
        legend_lines = [line for legend in canvas.figure.legends for line in legend.get_lines()]

        for child in frame_style.winfo_children():
            child.destroy()

        cvs = {}
        svs = {}
        for le in legend_lines:
            name = le.get_label()
            dc = 'C0'
            ds = '-'
            dc = le.get_color()
            ds = le.get_linestyle()
            ds = le.get_marker() if ds == 'None' else ds
            color_var, style_var = plot_style_widget(frame_style, text=name,
                                                     command=update_plot_style, padding=padding_setting,
                                                     default_color=dc, default_style=ds)
            cvs[name] = color_var
            svs[name] = style_var
        color_vars.set(cvs)
        style_vars.set(svs)
        return

    def update_plot_contents(canvas, name_to_key, *args, **common_kwargs):

        def no_data():
            canvas.figure.clear()
            canvas.figure.text(0.5, 0.5, 'Not calculated for this pair',
                               horizontalalignment='center', verticalalignment='center')
            return
        # canvas.figure.clear()
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
        if t in ['Compare Raw', 'Matched Peaks']:
            data2 = data_dict[name_to_key[n2]]

        if t in ['Raw Data', 'Peak Fits']:
            data_selection2_label.pack_forget()
            data_selection2_menu.pack_forget()
            custom_plot_action_label.pack_forget()
            custom_plot_action_menu.pack_forget()
        elif t in ['Compare Raw', 'Matched Peaks']:
            data_selection2_label.pack(side=tk.LEFT)
            data_selection2_menu.pack(side=tk.LEFT)
            custom_plot_action_label.pack_forget()
            custom_plot_action_menu.pack_forget()
        elif t in ['Custom Plot']:
            data_selection2_label.pack_forget()
            data_selection2_menu.pack_forget()
            custom_plot_action_label.pack(side=tk.LEFT)
            custom_plot_action_menu.pack(side=tk.LEFT)

        # Make Plots
        if t == 'Raw Data':
            x, y = data.freq.copy()/1000, data.vel.copy()
            _ = line_plot(x, y, legend=[n], x_label='Frequency (kHz)', y_label='Intensity (µm/s)',
                          title='Raw Data', **kwargs)
        elif t == 'Peak Fits':
            data_selection2_label.pack_forget()
            data_selection2_menu.pack_forget()
            x, y, p = data.freq.copy()/1000, data.newvel.copy(), data.peaks['positions']/1000
            f0, f1 = common_kwargs['x_lim']
            x = x[np.logical_and(x > f0, x < f1)]
            _ = line_plot(x, y, legend=[n], x_label='Frequency (kHz)', y_label='Intensity (arb.)',
                          v_line_pos=p, v_line_width=2, v_line_legend=['Peak Locations'],
                          y_norm='each', title='Peak Fits',
                          legend_location='upper right', **kwargs)
        elif t == 'Compare Raw':
            data_selection2_label.pack(side=tk.LEFT)
            data_selection2_menu.pack(side=tk.LEFT)
            x, y = data.freq.copy()/1000, data.vel.copy()
            x2, y2 = data2.freq.copy()/1000, data2.vel.copy()
            _ = line_plot([x, x2], [y, y2], x_label='Frequency (kHz)', y_label='Intensity (µm/s)',
                          legend=[n, n2], legend_location='upper right', title='Peak Fits', **kwargs)
        elif t == 'Matched Peaks':
            data_selection2_label.pack(side=tk.LEFT)
            data_selection2_menu.pack(side=tk.LEFT)
            pr = [pair for pair in pair_results if n in pair['names'] and n2 in pair['names']]
            if not pr:
                no_data()
            else:
                pr = pr[0]
                s = pr['stretch']
                x, y = data.freq.copy()/1000, data.newvel.copy()
                x2, y2 = data2.freq.copy()/1000, data2.newvel.copy()

                f0, f1 = common_kwargs['x_lim']
                x = x[np.logical_and(x > f0, x < f1)]
                x2 = x2[np.logical_and(x2 > f0, x2 < f1)]

                if n == pr['names'][0]:
                    x2 *= s
                    unmatched = pr['unmatched']
                else:
                    x *= s
                    unmatched = pr['unmatched'][::-1]

                _ = line_plot([x, x2], [y, y2], x_label='Frequency (kHz)', y_label='Normalized Intensity (arb.)',
                              v_line_pos=[pr['matched'], *unmatched],
                              v_line_color=['k', 'C0', 'C1'], v_line_width=[4, 2, 2],
                              legend=[n, n2], v_line_legend=['Matched', f'Unmatched {n}', f'Unmatched {n2}'],
                              legend_location='upper right', y_norm='each',
                              title='Stretched peak matches raw', **kwargs)
        elif t == 'Custom Plot':
            action = custom_plot_action_var.get()
            if action == 'Clear All':
                if (
                    len(canvas.figure.get_children()) != 1
                    and (
                        custom_plot_clear_var.get()
                        or tk.messagebox.askokcancel('Clear Plots?',
                                                     'Are you sure you want to completely clear the plot?',
                                                     parent=window, master=window
                                                     )
                    )
                ):
                    canvas.figure.clear()
            elif 'Add' in action:
                x, p = data.freq.copy()/1000, data.peaks['positions']/1000
                y = data.newvel.copy() if 'Smoothed' in action else data.vel.copy()
                f0, f1 = common_kwargs['x_lim']
                x = x[np.logical_and(x > f0, x < f1)] if 'Smoothed' in action else x

                if 'with Peaks' in action:
                    vline_kwargs = {'v_line_pos': p, 'v_line_width': 2, 'v_line_legend': [n+' Peaks']}
                else:
                    vline_kwargs = {}

                line_plot(x, y, legend=[n + ' Smoothed' if 'Smoothed' in action else n],
                          x_label='Frequency (kHz)', y_label='Intensity (arb.)',
                          title='Custom Plot',
                          legend_location='upper right', clear_fig=False, **kwargs, **vline_kwargs)

                pass
            elif 'Remove' in action:
                removed_lines = False
                lines = [line for axis in canvas.figure.axes for line in axis.get_lines()]
                line_names = [line.get_label() for line in lines]
                for line, name in zip(lines, line_names):
                    if action == 'Remove':
                        if name in [n, '_'+n, n+' Peaks', '_'+n+' Peaks', n+' Smoothed', '_'+n+' Smoothed']:
                            line.remove()
                            removed_lines = True
                    elif action == 'Remove Peaks':
                        for n in name_to_key:
                            if name in [n+' Peaks', '_'+n+' Peaks']:
                                line.remove()
                                removed_lines = True

                if removed_lines:
                    # Update Legend
                    leglines = [line for legend in canvas.figure.legends for line in legend.get_lines()]
                    legline_names = [line.get_label() for line in leglines]
                    for legend in canvas.figure.legends:
                        legend.remove()
                    for line, name in zip(reversed(leglines), reversed(legline_names)):
                        if action == 'Remove':
                            if name in [n, '_'+n, n+' Peaks', '_'+n+' Peaks', n+' Smoothed', '_'+n+' Smoothed']:
                                leglines.remove(line)
                                legline_names.remove(name)
                        elif action == 'Remove Peaks':
                            for n in name_to_key:
                                if name in [n+' Peaks', '_'+n+' Peaks']:
                                    leglines.remove(line)
                                    legline_names.remove(name)
                    leg = canvas.figure.legend(leglines, legline_names, loc='upper right', framealpha=1, fancybox=False)
                    leg.get_frame().set_linewidth(4)
                    leg.get_frame().set_edgecolor('black')
                    make_legend_interactive(canvas.figure, leg, lines)

        # required to update canvas and attached toolbar!
        canvas.draw_idle()
        update_style_options()
        return

    frame_options_legend = tk.Frame(window)
    frame_options = tk.Frame(frame_options_legend)
    frame_style = tk.Frame(frame_options_legend)

    plot_type_options = ['Raw Data', 'Peak Fits', 'Compare Raw', 'Matched Peaks', 'Custom Plot']
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
        name_to_key = {}
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

    custom_plot_options = ['Add Raw', 'Add Smoothed', 'Add Raw with Peaks', 'Add Smoothed with Peaks',
                           'Remove', 'Remove Peaks', 'Clear All']
    custom_plot_action_var = tk.StringVar(window)
    custom_plot_clear_var = tk.BooleanVar(window)
    custom_plot_action_label = tk.Label(frame_options, text="Plotting Action")
    custom_plot_action_menu = ttk.OptionMenu(frame_options, custom_plot_action_var,
                                             custom_plot_options[0], *custom_plot_options)

    def update_plot_contents_wrapper(*args):
        if args[0] == data_selection_var._name and plot_type_var.get() == 'Custom Plot':
            return
        if args[0] == plot_type_var._name and plot_type_var.get() == 'Custom Plot':
            custom_plot_clear_var.set(True)
            custom_plot_action_var.set('Clear All')
            custom_plot_clear_var.set(False)
            return

        data_selection2_var.trace_remove("write", data_selection2_var_traceid_var.get())
        update_plot_contents(canvas, name_to_key, **common_kwargs)
        traceid = data_selection2_var.trace_add("write", update_plot_contents_wrapper)
        data_selection2_var_traceid_var.set(traceid)
        return

    data_selection2_var_traceid_var = tk.StringVar()
    plot_type_var.trace_add("write", update_plot_contents_wrapper)
    data_selection_var.trace_add("write", update_plot_contents_wrapper)
    traceid = data_selection2_var.trace_add("write", update_plot_contents_wrapper)
    data_selection2_var_traceid_var.set(traceid)
    custom_plot_action_var.trace_add("write", update_plot_contents_wrapper)

    plot_type_label.pack(side=tk.LEFT)
    plot_type_menu.pack(side=tk.LEFT)
    data_selection_label.pack(side=tk.LEFT)
    data_selection_menu.pack(side=tk.LEFT)

    frame_options_legend.pack(side=tk.BOTTOM)
    frame_options.pack(side=tk.TOP)
    frame_style.pack(side=tk.TOP)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    update_plot_contents(canvas, name_to_key, **common_kwargs)

    return


if __name__ == '__main__':
    from app import run_app
    run_app()
