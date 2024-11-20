# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:08:28 2024

@author: KOlson
"""
import tkinter as tk

try:
    from app_helpers import labeled_options, make_button
except ModuleNotFoundError:
    from MetroLaserLARS.app_helpers import labeled_options, make_button


def open_part_matching_window(root, grouped_folders_var, part_matching_text_var, part_matching_strategy_var, **common_kwargs):
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

    window = tk.Toplevel(root)
    window.grab_set()
    window.title("Define Known Part Matching")
    label1 = tk.Label(window,
                      text="""
Define known part matching by entering a list of equivalent parts,
by creating a custom function, or by the folder structure (if "Use
grouped folder structure" is set to True).""")
    label1.pack()
    options = ['folder', 'list', 'custom'] if grouped_folders_var.get() == 'True' else ['list', 'custom']
    _, _, _, _, _, _ = labeled_options(window, 'Definition type:',
                                       var=part_matching_strategy_var,
                                       options=options,
                                       infobox=False, **common_kwargs)

    frame_input = tk.Frame(window)
    frame_input.pack()

    code_label_1 = tk.Label(frame_input, text='def part_matching_function(name0, name1):', font=("Courier New", 9))
    code_label_1.pack(anchor="w")

    frame_text = tk.Frame(frame_input)
    frame_text.pack()

    code_label_2 = tk.Label(frame_text, text='    ', font=("Courier New", 9))
    code_label_2.pack(side=tk.LEFT)

    part_matching_text = tk.Text(frame_text, bg='white')
    part_matching_text.insert("0.0", part_matching_text_var.get())
    part_matching_text.pack(side=tk.LEFT)

    code_label_3 = tk.Label(frame_input, text='return result', font=("Courier New", 9))
    code_label_3.pack(anchor="w")

    make_button(window, text='Save entered text',
                command=lambda: part_matching_text_var.set(
                    part_matching_text.get("0.0", tk.END)), side=tk.TOP)

    fn = toggle_code_labels
    part_matching_strategy_var.trace_add('write', fn)
    toggle_code_labels()

    return part_matching_text_var.get()


if __name__ == '__main__':
    from app import run_app
    run_app()
