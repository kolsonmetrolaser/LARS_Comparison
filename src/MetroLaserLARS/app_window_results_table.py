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
    from app_helpers import labeled_options, labeled_entry, padding_setting, bool_options
    from app_helpers import background_color as bgc
except ModuleNotFoundError:
    from MetroLaserLARS.app_helpers import labeled_options, labeled_entry, padding_setting, bool_options
    from MetroLaserLARS.app_helpers import background_color as bgc


def get_curr_screen_geometry():
    """
    Workaround to get the size of the current screen in a multi-screen setup.

    Returns:
        geometry (str): The standard Tk geometry string.
            [width]x[height]+[left]+[top]
    """
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_width(), root.winfo_height()
    root.destroy()
    return geometry


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

    def fill_table(canvas, varroot, data: str = None, data_dict: dict = {}, pair_results: list[dict] = [{}],
                   font_size=None):
        if not data_dict or not pair_results:
            return

        reference = reference_var.get()
        reference = '' if reference == '(Full Table)' else reference

        canvas.delete('all')

        default_font_name = tk.font.nametofont('TkTextFont').actual()['family']

        if data is None:
            data = table_options_var.get()
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

        if font_size is None:
            canvas.update()
            sizelimit = min(canvas.winfo_width()/5/(len(data_dict)+2), canvas.winfo_height()/2/(len(data_dict)+2))
            size = min(sizelimit, 1/48*min(get_curr_screen_geometry()))
            font_size = int(12/10*size)
            font_size_var.set(font_size)
        else:
            font_size = int(font_size)

        for i, d1 in enumerate(data_dict.values()):
            if reference:
                if d1.name != reference:
                    continue
                i = 0
            default_font_name = ''.join(default_font_name.split())
            size = 10/12*font_size
            canvas.create_text(5*size, 2*size*(i+3), text=d1.name, font=f"{default_font_name} {int(font_size)}")
            for j, d2 in enumerate(data_dict.values()):
                canvas.create_text(5*size*(j+2), 4*size, text=d2.name, font=f"{default_font_name} {int(font_size)}")
                same_part = 1. if d1.name == d2.name else (
                    np.array([
                        p for p in pair_results if d1.name in p['names'] and d2.name in p['names']
                    ][0]['same_part']).astype(dtype)
                )

                if data in ['matched']:
                    val = d1.peaks['count'] if d1.name == d2.name else len(
                        [p for p in pair_results if d1.name in p['names'] and d2.name in p['names']][0][data])
                elif data in ['unmatched']:
                    val = np.nan if d1.name == d2.name else len(
                        [f for ul in
                         [p for p in pair_results if d1.name in p['names'] and d2.name in p['names']][0][data]
                         for f in ul])
                elif data in ['same_part']:
                    val = copy(same_part)
                elif data in ['stretch']:
                    selfval = np.array(1).astype(dtype)
                    val = selfval if d1.name == d2.name else (
                        np.array([
                            p for p in pair_results if d1.name in p['names'] and d2.name in p['names']
                        ][0][data]).astype(dtype)
                    )
                    val = val if i >= j else 1/val
                else:
                    if data in ['match_probability']:
                        selfval = np.array(1).astype(dtype)
                    elif data in ['quality']:
                        selfval = np.nan
                    val = selfval if d1.name == d2.name else (
                        np.array([
                            p for p in pair_results if d1.name in p['names'] and d2.name in p['names']
                        ][0][data]).astype(dtype)
                    )

                textvars[i][j] = tk.StringVar(varroot, value='' if np.isnan(val) else eval("f'{val"+printformat+"}'"))
                cmap_val = ((val-min_val)/(max_val-min_val) if data not in ['stretch'] else
                            (np.abs(val-1)/max(np.abs(max_val-1), np.abs(min_val-1))))
                canvas.create_rectangle(5*size*(j+1.5), 2*size*(i+2.5), 5*size*(j+2.5), 2*size*(i+3.5),
                                        fill=_from_cmap(_cmap(
                                            cmap_val if data not in ['stretch'] else
                                            (1 - (1 + (((1-cmap_val)*lp)/(cmap_val*(1-lp)))**2)**-1
                                             if cmap_val != 0 and cmap_val != 1 else 1-cmap_val
                                             )
                                        )))
                fontstr = f"{default_font_name} {int(font_size)}{' bold' if d1.name == d2.name else ''}"
                canvas.create_text(5*size*(j+2), 2*size*(i+3), text=textvars[i][j].get(),
                                   font=fontstr)
        fontstr = f"{default_font_name} {int(font_size)} bold"
        canvas.create_text(1/2*5*size, 1/2*2*size*(len(data_dict)+3) if not reference else 5.5*size,
                           text='Reference', angle=90, font=fontstr)
        canvas.create_text(1/2*5*size*(len(data_dict)+2), 1/2*4*size,
                           text='Measurement', font=fontstr)

    window = tk.Toplevel(root, bg=bgc)
    window.title("Data Tables")
    window.geometry("1600x900")

    data_dict, pair_results = data_dict_var.get(), pair_results_var.get()
    data_dict = sort_data_dict(data_dict, pair_results)

    frame_options = tk.Frame(window, bg=bgc)
    frame_options.pack()

    frame_sheet = tk.Frame(window, bg=bgc)
    frame_sheet.pack(expand=True, fill=tk.BOTH)
    canvas_sheet = tk.Canvas(frame_sheet, bg=bgc)
    canvas_sheet.pack(expand=True, fill=tk.BOTH)

    # font_size_var, _, _, _, _, _ = labeled_entry(frame_options, 'Font Size:', padding=padding_setting, side=tk.LEFT,
    #                                               vardefault=tk.font.nametofont('TkTextFont').actual()['size'],
    #                                               vartype=tk.StringVar, infobox=False,
    #                                               command=lambda *args: fill_table(frame_sheet, root,
    #                                                                               data_dict=data_dict,
    #                                                                               pair_results=pair_results,
    #                                                                               font_size=font_size_var.get()))
    font_size_var, _, _, font_size_entry, _, _ = labeled_entry(frame_options, 'Font Size:', padding=padding_setting,
                                                               side=tk.LEFT,
                                                               vardefault=tk.font.nametofont('TkTextFont').actual()['size'],
                                                               vartype=tk.StringVar, infobox=False)
    font_size_entry.bind('<FocusOut>', lambda *args: fill_table(canvas_sheet, root,
                                                                data_dict=data_dict,
                                                                pair_results=pair_results,
                                                                font_size=font_size_var.get()))

    table_options = ['Match Probability', 'Matching Quality', 'Stretching Factor', 'Number of Matched Peaks', 'Number of Unmatched Peaks', 'Same Part']
    table_options_var, _, _, _, _, _ = labeled_options(frame_options, 'Data to compare:', padding=padding_setting,
                                                       vartype=tk.StringVar, vardefault=table_options[0], side=tk.LEFT,
                                                       command=lambda *args: fill_table(canvas_sheet, root,
                                                                                        data_dict=data_dict,
                                                                                        pair_results=pair_results),
                                                       infobox=False, options=table_options)

    reference_options = ['(Full Table)']+[d.name for d in data_dict.values()]
    reference_var, _, _, _, _, _ = labeled_options(frame_options, 'Choose Reference:', padding=padding_setting,
                                                   vartype=tk.StringVar, vardefault=reference_options[0], side=tk.LEFT,
                                                   command=lambda *args: fill_table(canvas_sheet, root,
                                                                                    data_dict=data_dict,
                                                                                    pair_results=pair_results),
                                                   infobox=False, options=reference_options)

    fill_table(canvas_sheet, root, data=table_options[0], data_dict=data_dict, pair_results=pair_results)
    window.bind_all("<1>", lambda event: event.widget.focus_set())

    return


if __name__ == '__main__':
    from app import run_app
    run_app()