# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:52:07 2024

@author: KOlson
"""
import tkinter as tk
import numpy as np
import os.path as osp
from copy import copy

# Internal imports
try:
    from app_helpers import labeled_options, make_button
    from app_helpers import make_window
except ModuleNotFoundError:
    from MetroLaserLARS.app_helpers import labeled_options, make_button  # type: ignore
    from MetroLaserLARS.app_helpers import make_window  # type: ignore


def open_peak_list_window(root, data_dict_var, directory_var):

    window = make_window(root, "Peak List", (800, 450))

    def write_text(*_):
        if not data_dict:
            return

        peak_list_text.delete('1.0', tk.END)
        peaks = copy(data_dict[name_to_key[part_selection_var.get()]].peaks['positions'])
        if units_var.get() == 'kHz':
            peaks /= 1000

        format_to_character = {'column': '\n',
                               'tab separated row': '\t',
                               'comma separated row': ','}
        text = format_to_character[format_var.get()].join([f'{p:.7g}' for p in peaks])
        peak_list_text.insert("0.0", text)
        return

    options_frame = tk.Frame(window)
    options_frame.pack(side=tk.TOP)
    peak_list_text = tk.Text(window, bg='white')
    peak_list_text.insert("0.0", '')
    peak_list_text.pack(side=tk.TOP, expand=True, fill='both')

    data_dict = data_dict_var.get()

    data_keys = list(data_dict.keys())
    if data_keys:
        name_to_key = {}
        data_options = []
        for k in data_keys:
            data_options.append(data_dict[k].name)
            name_to_key[data_dict[k].name] = k
    else:
        name_to_key = {}
        data_options = ['']

    part_selection_var, _, _, _, _, _ = labeled_options(options_frame, 'Part:', vartype=tk.StringVar,
                                                        options=data_options, vardefault=data_options[0],
                                                        infotext='The part for which to list peaks.',
                                                        side=tk.LEFT, command=write_text)
    format_var, _, _, _, _, _ = labeled_options(options_frame, 'Format:', vartype=tk.StringVar,
                                                options=['column', 'tab separated row', 'comma separated row'],
                                                vardefault='column',
                                                infotext='The list of peaks will be displayed in the selected format.',
                                                side=tk.LEFT, command=write_text)
    units_var, _, _, _, _, _ = labeled_options(options_frame, 'Units:', vartype=tk.StringVar,
                                               options=['Hz', 'kHz'], vardefault='Hz',
                                               infotext='The units of the peak list.',
                                               side=tk.LEFT, command=write_text)

    def save_peak_lists(*args):
        dd = data_dict_var.get()
        maxlen = 0
        for v in dd.values():
            maxlen = max(maxlen, len(v.peaks['positions']))
        dataout = np.nan*np.ones((len(dd), maxlen))
        header = ''
        for i, v in enumerate(dd.values()):
            dataout[i, :len(v.peaks['positions'])] = v.peaks['positions']
            header += v.name + ','
        header = header[:-1]
        # f = osp.join(pathlib.Path(list(dd.keys())[0]).parent, 'peak_list.csv')
        f = osp.join(directory_var.get(), 'peak_list.csv')
        np.savetxt(f, dataout.T, delimiter=',', header=header)
        print(f'Saved peak list to {f}')

    make_button(options_frame, 'Save Peak Lists', command=save_peak_lists,
                side=tk.LEFT, infobox=True,
                infotext="""Saves the peak lists in a .csv so that it
can be used in other programs (e.g., Excel)""")

    write_text()
    return


if __name__ == '__main__':
    from app import run_app
    run_app()
