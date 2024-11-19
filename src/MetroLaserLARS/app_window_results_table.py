# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:52:07 2024

@author: KOlson
"""
import tkinter as tk
import numpy as np

# Internal imports
try:
    from app_helpers import labeled_options, padding_setting, bool_options
except ModuleNotFoundError:
    from MetroLaserLARS.app_helpers import labeled_options, padding_setting, bool_options


def _from_cmap(c):
    r, g, b, a = c
    r = int(r*255)
    g = int(g*255)
    b = int(b*255)
    return f'#{r:02x}{g:02x}{b:02x}'


def _cmap(n):
    if np.isnan(n):
        return (1, 1, 1, 1)
    c = (0, 0, 0, 0)
    if n < 0:
        c = (1, 0, 0, 1)
    elif n <= 0.5:
        c = (1, n*2, 0, 1)
    elif n <= 1:
        c = ((1-n)*2, 1, 0, 1)
    elif n > 1:
        c = (0, 1, 0, 1)
    else:
        return (1, 1, 1, 1)
    w = 7/12
    c = tuple(w+(1-w)*el for el in c)
    return c[:3]+(1,)


def open_results_table_window(root, data_dict_var, pair_results_var, **common_kwargs):

    def fill_table(frame, varroot, data: str = None, data_dict: dict = {}, pair_results: list[dict] = [{}]):
        if data is None:
            data = save_data_var.get()
        input_to_internal = {'Match Probability': 'match_probability',
                             'Stretching Factor': 'stretch',
                             'Quality': 'quality',
                             'Same Part': 'same_part',
                             'Number of Matched Peaks': 'matched',
                             'Number of Unmatched Peaks': 'unmatched',
                             }
        data = input_to_internal[data] if data in input_to_internal else data

        grid_size = (len(data_dict)+1, len(data_dict)+1)
        textvars = [[[]]*grid_size[0]]*grid_size[1]

        if data in ['match_probability', 'stretch', 'same_part', 'quality']:
            vals = [p[data] for p in pair_results]
        elif data in ['matched']:
            vals = [len(p[data]) for p in pair_results]+[d.peaks['count'] for d in data_dict.values()]
        elif data in ['unmatched']:
            vals = [len([f for ul in p[data] for f in ul]) for p in pair_results]

        roundval = 3 if data in ['match_probability', 'stretch'] else 0
        dtype = np.float64
        printformat = ':.3f' if data in ['match_probability', 'stretch'] else ':.0f'

        min_val, max_val = np.min(vals).astype(dtype), np.max(vals).astype(dtype)

        for i, d1 in enumerate(data_dict.values()):
            for j, d2 in enumerate(data_dict.values()):
                if data in ['matched']:
                    val = d1.peaks['count'] if i == j else len(
                        [p for p in pair_results if d1.name in p['names'] and d2.name in p['names']][0][data])
                elif data in ['unmatched']:
                    val = np.nan if i == j else len(
                        [f for ul in
                         [p for p in pair_results if d1.name in p['names'] and d2.name in p['names']][0][data]
                         for f in ul])
                else:
                    if data in ['match_probability', 'stretch', 'same_part']:
                        selfval = np.array(1).astype(dtype)
                    elif data in ['quality']:
                        selfval = np.nan
                    val = selfval if i == j else (
                        np.array([
                            p for p in pair_results if d1.name in p['names'] and d2.name in p['names']
                        ][0][data]).astype(dtype)
                    )
                textvars[i][j] = tk.StringVar(varroot, value=eval("f'{val"+printformat+"}'"))
                entry = tk.Entry(frame, textvariable=textvars[i][j],
                                 bg=_from_cmap(
                    _cmap(
                        (val-min_val)/(max_val-min_val)
                    )
                )
                )
                entry.grid(row=i+1, column=j+1)

    window = tk.Toplevel(root)
    window.title("Plots")
    window.geometry("1600x900")

    data_dict, pair_results = data_dict_var.get(), pair_results_var.get()

    frame_options = tk.Frame(window)
    frame_options.pack()

    frame_sheet = tk.Frame(window)
    frame_sheet.pack()

    for i, (k1, d1) in enumerate(data_dict.items()):
        labelr = tk.Label(frame_sheet, text=d1.name)
        labelc = tk.Label(frame_sheet, text=d1.name)
        labelr.grid(row=i+1, column=0)
        labelc.grid(row=0, column=i+1)

    # fill_table(frame_sheet, root, data='match_probability', data_dict=data_dict, pair_results=pair_results)

    table_options = ['Match Probability', 'Stretching Factor', 'Quality', 'Number of Matched Peaks', 'Number of Unmatched Peaks', 'Same Part']
    save_data_var, _, _, _, _, _ = labeled_options(frame_options, 'Data to compare:', padding=padding_setting,
                                                   vartype=tk.StringVar, vardefault=table_options[0],
                                                   command=lambda event: fill_table(frame_sheet, root,
                                                                                    data_dict=data_dict,
                                                                                    pair_results=pair_results),
                                                   infobox=False, options=table_options)

    return


if __name__ == '__main__':
    from app import run_app
    run_app()
