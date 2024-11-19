# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:52:07 2024

@author: KOlson
"""
import tkinter as tk
import numpy as np
from copy import copy

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


def sort_data_dict(data_dict, pair_results):
    sorted_data_dict = {}

    while len(sorted_data_dict.keys()) < len(data_dict.keys()):
        for k, d in data_dict.items():
            if k not in sorted_data_dict:
                sorted_data_dict[k] = d
                for k2, d2 in data_dict.items():
                    if [p for p in pair_results if d.name in p['names'] and d2.name in p['names']][0]['same_part']:
                        sorted_data_dict[k2] = d2

    return sorted_data_dict


def open_results_table_window(root, data_dict_var, pair_results_var, **common_kwargs):

    def fill_table(frame, varroot, data: str = None, data_dict: dict = {}, pair_results: list[dict] = [{}]):
        default_font_name = tk.font.nametofont('TkTextFont').actual()['family']
        default_font_size = tk.font.nametofont('TkTextFont').actual()['size']

        if data is None:
            data = save_data_var.get()
        input_to_internal = {'Match Probability': 'match_probability',
                             'Stretching Factor': 'stretch',
                             'Matching Quality': 'quality',
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

        if data in ['stretch']:
            lp = np.median(np.abs(1-np.array(vals))/max(np.abs(1-min_val), np.abs(1-max_val)))

        for i, d1 in enumerate(data_dict.values()):
            for j, d2 in enumerate(data_dict.values()):
                same_part = 1. if i == j else (
                    np.array([
                        p for p in pair_results if d1.name in p['names'] and d2.name in p['names']
                    ][0]['same_part']).astype(dtype)
                )

                if data in ['matched']:
                    val = d1.peaks['count'] if i == j else len(
                        [p for p in pair_results if d1.name in p['names'] and d2.name in p['names']][0][data])
                elif data in ['unmatched']:
                    val = np.nan if i == j else len(
                        [f for ul in
                         [p for p in pair_results if d1.name in p['names'] and d2.name in p['names']][0][data]
                         for f in ul])
                elif data in ['same_part']:
                    val = copy(same_part)
                else:
                    if data in ['match_probability', 'stretch']:
                        selfval = np.array(1).astype(dtype)
                    elif data in ['quality']:
                        selfval = np.nan
                    val = selfval if i == j else (
                        np.array([
                            p for p in pair_results if d1.name in p['names'] and d2.name in p['names']
                        ][0][data]).astype(dtype)
                    )

                textvars[i][j] = tk.StringVar(varroot, value='' if np.isnan(val) else eval("f'{val"+printformat+"}'"))
                cmap_val = ((val-min_val)/(max_val-min_val) if data not in ['stretch'] else
                            (np.abs(val-1)/max(np.abs(max_val-1), np.abs(min_val-1))))
                entry = tk.Entry(frame, textvariable=textvars[i][j], width=5,
                                 bg=_from_cmap(_cmap(
                                     cmap_val if data not in ['stretch'] else
                                     (1 - (1 + (((1-cmap_val)*lp)/(cmap_val*(1-lp)))**2)**-1
                                      if cmap_val != 0 and cmap_val != 1 else 1-cmap_val
                                      )
                                 )),
                                 font=(default_font_name, default_font_size, 'bold' if same_part else 'normal'))
                entry.grid(row=i+1, column=j+1)

    window = tk.Toplevel(root)
    window.title("Plots")
    window.geometry("1600x900")

    data_dict, pair_results = data_dict_var.get(), pair_results_var.get()
    data_dict = sort_data_dict(data_dict, pair_results)

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

    table_options = ['Match Probability', 'Matching Quality', 'Stretching Factor', 'Number of Matched Peaks', 'Number of Unmatched Peaks', 'Same Part']
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
