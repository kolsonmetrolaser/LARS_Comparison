# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:58:07 2024

@author: KOlson
"""
import tkinter as tk

if __name__ == '__main__':
    pass
else:
    pass

bool_options = ['True', 'False']
padding_heading = {'pady': 10, 'padx': 10}
padding_setting = {'pady': 4, 'padx': 4}
padding_option = {'pady': 0, 'padx': 4}
padding_none = {'pady': 0, 'padx': 0}
kwargs_pickle = {'title': "Select a data_dict[...].pkl file",
                 'filetypes': [("Pickled Data Dictionaries", "data_dict*.pkl"), ("All Files", "*.*")]}


class CustomVar(tk.Variable):
    def __init__(self):
        super().__init__()
        self.current = None

    def set(self, value: object):
        self.current = value
        super().set(value)

    def get(self):
        super().get()
        return self.current


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
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
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


def select_directory(entry):
    directory = tk.filedialog.askdirectory(title="Select a Directory")
    if directory:
        entry.delete(0, tk.END)  # Clear the directory_entry box
        entry.insert(0, directory)  # Insert the selected directory


def select_file(entry, **kwargs):
    path = tk.filedialog.askopenfilename(**kwargs)
    if path:
        entry.delete(0, tk.END)  # Clear the pickled_data_path_entry box
        entry.insert(0, path)  # Insert the selected pickled_data_path


def heading(txt, frame, lvl=0, padding=True, side=tk.TOP, subtext=None):
    default_font_name = tk.font.nametofont('TkTextFont').actual()['family']
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
        label = tk.Label(frame, text=text)
        label.pack(side=tk.LEFT)
        if text == '(?)':
            f = tk.font.Font(label, label.cget("font"))
            f.configure(underline=True)
            label.configure(font=f)
    return label


def labeled_entry(baseframe, label: str = '', varframe=None, postlabel: str = '', padding=padding_setting,
                  entry_width: int = 6, vardefault=0, vartype=None, update_status=None,
                  side=tk.TOP, infobox=True, infotext='Placeholder info text.'):
    varframe = baseframe if varframe is None else varframe
    entry, infolabel = None, None
    frame = labeled_widget_frame(baseframe, padding, side)
    label1 = labeled_widget_label(frame, label)
    if vartype is not None:
        var = vartype(varframe, value=vardefault)
        if update_status is not None:
            var.trace_add("write", update_status)
        entry = tk.Entry(frame, width=entry_width, textvariable=var)
        entry.pack(side=tk.LEFT)
    label2 = labeled_widget_label(frame, postlabel)
    if infobox:
        infolabel = labeled_widget_label(frame, '(?)')
        CreateToolTip(infolabel, infotext)
    return var, frame, label1, entry, label2, infolabel


def labeled_options(baseframe, label: str = '', varframe=None, postlabel: str = '', padding=padding_setting,
                    var=None, vardefault=None, vartype=None, update_status=None,
                    command=None, side=tk.TOP, options=bool_options, infobox=True,
                    infotext='Placeholder info text.'):
    varframe = baseframe if varframe is None else varframe
    optionmenu, infolabel = None, None
    frame = labeled_widget_frame(baseframe, padding, side)
    label1 = labeled_widget_label(frame, label)
    if vartype is not None or var is not None:
        if var is not None and vardefault is not None:
            var.set(vardefault)
        if var is None:
            var = vartype(varframe, value=vardefault)
        if update_status is not None:
            var.trace_add("write", update_status)
        optionmenu = tk.OptionMenu(frame, var, *options, command=command)
        optionmenu.config(bg='gray75', highlightthickness=0)
        optionmenu.pack(side=tk.LEFT)
    label2 = labeled_widget_label(frame, postlabel)
    if infobox:
        infolabel = labeled_widget_label(frame, '(?)')
        CreateToolTip(infolabel, infotext)
    return var, frame, label1, optionmenu, label2, infolabel


def labeled_file_select(baseframe, headingtxt: str = '', varframe=None, subheading: str = '', label: str = '',
                        padding=padding_setting, entry_width: int = 40, vardefault='', vartype=tk.StringVar,
                        update_status=None, command=None, side=tk.TOP, selection='file',
                        filetype='pickle', infobox=True, infotext='Placeholder info text.'):
    varframe = baseframe if varframe is None else varframe
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
        if update_status is not None:
            var.trace_add("write", update_status)
        entry = tk.Entry(frame, width=entry_width, textvariable=var)
        if selection == 'file':
            fun = select_file
            if filetype == 'pickle':
                kwargs = kwargs_pickle
        elif selection == 'dir':
            kwargs = {}
            fun = select_directory
        button = tk.Button(frame, text="Open", command=lambda: fun(entry, **kwargs), bg='gray75')
        button.pack(side=tk.LEFT, padx=4)
        entry.pack(side=tk.LEFT, padx=4)
        if infobox:
            infolabel = labeled_widget_label(frame, '(?)')
            CreateToolTip(infolabel, infotext)
    return var, frame, label1, entry, button, infolabel
