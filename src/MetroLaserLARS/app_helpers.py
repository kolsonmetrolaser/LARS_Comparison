# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:58:07 2024

@author: KOlson
"""
import os.path as osp
import sys
import tkinter as tk
from matplotlib.colors import to_hex, to_rgb

bool_options = ['True', 'False']
padding_heading = {'pady': 10, 'padx': 10}
padding_setting = {'pady': 4, 'padx': 4}
padding_option = {'pady': 0, 'padx': 4}
padding_none = {'pady': 0, 'padx': 0}
filetype_kwargs = {'pickle': {'title': "Select a data_dict[...].pkl file",
                              'filetypes': [("Pickled Data Dictionaries", "data_dict*.pkl"), ("All Files", "*.*")]},
                   'ml_weights': {'title': "Select a weights file",
                                  'filetypes': [("weights", "*.h5"), ("All Files", "*.*")]}
                   }

background_color = '#FFFEFC'
entry_color = 'white'
button_color = 'gray92'
active_bg = 'floral white'
active_fg = 'dark goldenrod'
options_style = ['-', ':', '--', '-.', '.', 'o', 'v', '^', '<', '>', 's', '*', 'd', 'p', 'x']
options_color = ['C'+str(i) for i in range(10)]


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = osp.abspath('.')
    return osp.join(base_path, relative_path)


icon_ML = resource_path('MLicon_128.png')


def clone_widget(widget, master=None):
    """
    Create a cloned version of a widget

    Parameters
    ----------
    widget : tkinter widget
        tkinter widget that shall be cloned.
    master : tkinter widget, optional
        Master widget onto which cloned widget shall be placed. If None, same master of input widget will be used. The
        default is None.

    Returns
    -------
    cloned : tkinter widget
        Clone of input widget onto master widget.

    """
    # Get main info
    parent = master if master else widget.master
    cls = widget.__class__

    # Clone the widget configuration
    cfg = {key: widget.cget(key) for key in widget.configure()}
    cloned = cls(parent, **cfg)

    # Clone the widget's children
    for child in widget.winfo_children():
        child_cloned = clone_widget(child, master=cloned)
        if child.grid_info():
            grid_info = {k: v for k, v in child.grid_info().items() if k not in {'in'}}
            child_cloned.grid(**grid_info)
        elif child.place_info():
            place_info = {k: v for k, v in child.place_info().items() if k not in {'in'}}
            child_cloned.place(**place_info)
        else:
            pack_info = {k: v for k, v in child.pack_info().items() if k not in {'in'}}
            child_cloned.pack(**pack_info)

    return cloned


def log_decorator(func, log_var, log_file_loc_var, running_var):
    def inner(inputStr):
        try:
            new_text = log_var.get() + inputStr
            if not running_var.get() or len(new_text) > 100000:
                with open(log_file_loc_var.get(), 'a', encoding="utf-8") as f:
                    f.write(new_text)
                    log_var.set('')
            else:
                log_var.set(log_var.get()+inputStr)
            return func(inputStr)
        except Exception:
            print('Error inside log_decorator, doing default print')
            return func(inputStr)
    return inner


def make_window(parent, title: str = '', size: tuple[int, int] | None = None):
    window = tk.Toplevel(parent, bg=background_color)
    window.title(title)
    if size is not None:
        window.geometry(str(size[0])+"x"+str(size[1]))
    window.wm_iconphoto(False, tk.PhotoImage(file=icon_ML))
    window.focus_force()
    edit_name_menu_bar(window)
    return window


def open_log_window(root, log_var, log_file_loc_var):
    window = make_window(root, "Log", (800, 450))
    print('opened log window')

    log_text = tk.Text(window, bg='white')
    with open(log_file_loc_var.get(), 'r') as f:
        text_from_log_file = f.read()
    log_text.insert("0.0", text_from_log_file + log_var.get())
    log_text.pack(side=tk.LEFT, expand=True, fill='both')

    def update_log(*_):
        log_text.delete('1.0', tk.END)
        log_text.insert("0.0", log_var.get())
        return

    traceid = log_var.trace_add("write", update_log)

    def on_closing():
        log_var.trace_remove("write", traceid)
        window.destroy()

    window.protocol('WM_DELETE_WINDOW', on_closing)

    return


def unique(mylist):
    return list(dict.fromkeys(mylist))


def edit_name_menu_bar(window):
    menubar = tk.Menu(window)
    menubar.add_command(label="Rename Window",
                        command=lambda: window.title(
                            tk.simpledialog.askstring(title='Rename window',
                                                      prompt='Enter new window name:',
                                                      initialvalue=window.title()
                                                      ))
                        )
    window.config(menu=menubar)
    return


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
        self.text = ''

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


def labeled_widget_frame(baseframe, padding, side, grid):
    lwframe = tk.Frame(baseframe)
    if grid is None:
        lwframe.pack(**padding, side=side)
    else:
        lwframe.pack(row=grid[0], column=grid[1], **padding)
    return lwframe


def labeled_widget_label(frame, text, side=tk.LEFT):
    label = tk.Label(frame, text=text)
    label.pack(side=side)
    if text != '':
        if text == '(?)':
            f = tk.font.Font(label, label.cget("font"))
            f.configure(underline=True)
            label.configure(font=f)
    return label


def labeled_entry(baseframe, label: str = '', varframe=None, postlabel: str = '', padding=padding_setting,
                  var=None, entry_width: int = 6, vardefault=None, vartype=None, update_status=None, command=None,
                  side=tk.TOP, grid=None, infobox=True, infotext='Placeholder info text.'):
    if command is not None and update_status is not None:
        raise Exception("Only one of command and update_status may be specified")
    update_status = command if update_status is None else update_status
    varframe = baseframe if varframe is None else varframe
    entry, infolabel = None, None
    frame = labeled_widget_frame(baseframe, padding, side, grid)
    label1 = labeled_widget_label(frame, label)
    if vartype is not None or var is not None:
        if var is not None and vardefault is not None:
            var.set(vardefault)
        if var is None:
            var = vartype(varframe, value=vardefault)
        if update_status is not None:
            var.trace_add("write", update_status)
        entry = tk.Entry(frame, width=entry_width, textvariable=var, bg=entry_color)
        entry.pack(side=tk.LEFT)
    label2 = labeled_widget_label(frame, postlabel)
    if infobox:
        infolabel = labeled_widget_label(frame, '(?)')
        CreateToolTip(infolabel, infotext)
    return var, frame, label1, entry, label2, infolabel


def labeled_options(baseframe, label: str = '', varframe=None, postlabel: str = '', padding=padding_setting,
                    var=None, vardefault=None, vartype=None, update_status=None,
                    command=None, side=tk.TOP, options=bool_options, infobox=True,
                    infotext='Placeholder info text.', grid=None):
    varframe = baseframe if varframe is None else varframe
    optionmenu, infolabel = None, None
    frame = labeled_widget_frame(baseframe, padding, side, grid)
    label1 = labeled_widget_label(frame, label)
    if vartype is None and var is None:
        raise Exception('No variable or variable type provided.')
    if vartype is not None or var is not None:
        if var is not None and vardefault is not None:
            var.set(vardefault)
        if var is None:
            var = vartype(varframe, value=vardefault)
        if update_status is not None:
            var.trace_add("write", update_status)
        optionmenu = tk.OptionMenu(frame, var, *options, command=command)
        optionmenu.pack(side=tk.LEFT)
        optionmenu.config(bg=button_color, highlightthickness=0,
                          activebackground=active_bg, activeforeground=active_fg)
    label2 = labeled_widget_label(frame, postlabel)
    if infobox:
        infolabel = labeled_widget_label(frame, '(?)')
        CreateToolTip(infolabel, infotext)
    return var, frame, label1, optionmenu, label2, infolabel


def make_button(baseframe, text: str = '', command=None, padding=padding_setting,
                infobox=False, infotext='Placeholder info text.', side=tk.LEFT, grid=None):
    def update_button_color(event, *args, **kwargs):
        if event.type.name == 'Enter':
            event.widget['background'] = active_bg
            event.widget['foreground'] = active_fg
        if event.type.name == 'Leave':
            event.widget['background'] = button_color
            event.widget['foreground'] = 'black'

    frame = labeled_widget_frame(baseframe, padding, side, grid)

    b = tk.Button(frame, text=text,
                  command=command, bg=button_color)
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", update_button_color)
    b.bind("<Leave>", update_button_color)

    if infobox:
        infolabel = labeled_widget_label(frame, '(?)')
        CreateToolTip(infolabel, infotext)
    return b


def make_progress_bar(baseframe, var, text: str = '',
                      orient=tk.HORIZONTAL, length=200, mode="determinate",
                      maximum=1, padding=padding_setting, side=tk.LEFT, grid=None):
    frame = labeled_widget_frame(baseframe, padding, side, grid)
    labeled_widget_label(frame, text)

    progress_bar = tk.ttk.Progressbar(frame, orient=orient, length=length,
                                      mode=mode, maximum=maximum, takefocus=True)
    progress_bar['value'] = 0
    progress_bar.pack()

    def update_progress_bar(*args):
        progress_bar['value'] = var.get()
        progress_bar.update_idletasks()

    var.trace_add('write', update_progress_bar)
    return progress_bar


def open_progress_window(root, progress_vars, progress_texts, status_label):
    window = make_window(root, "Calculating...")

    for pvar, ptext in zip(progress_vars[::-1], progress_texts[::-1]):
        make_progress_bar(window, pvar, text=ptext, side=tk.BOTTOM)

    cw = clone_widget(status_label, window)
    cw.pack(side=tk.TOP)

    def update_cloned_status_label(*args):
        cw.config(**{key: status_label.cget(key) for key in status_label.configure()})
        cw.update()

    pvar.trace_add('write', update_cloned_status_label)
    return window


def labeled_file_select(baseframe, headingtxt: str = '', varframe=None, subheading: str = '', label: str = '',
                        padding=padding_setting, entry_width: int = 40, vardefault='', vartype=tk.StringVar,
                        update_status=None, side=tk.TOP, grid=None, selection='file',
                        filetype='pickle', infobox=True, infotext='Placeholder info text.'):
    varframe = baseframe if varframe is None else varframe
    entry, button, infolabel = None, None, None
    frame = labeled_widget_frame(baseframe, padding, side, grid)
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
        entry = tk.Entry(frame, width=entry_width, textvariable=var, bg=entry_color)
        if selection == 'file':
            fun = select_file
            kwargs = filetype_kwargs[filetype]
        elif selection == 'dir':
            kwargs = {}
            fun = select_directory
        button = make_button(frame, text="Open", command=lambda: fun(entry, **kwargs), padding={'padx': 4})
        entry.pack(side=tk.LEFT, padx=4)
        if infobox:
            infolabel = labeled_widget_label(frame, '(?)')
            CreateToolTip(infolabel, infotext)
    return var, frame, label1, entry, button, infolabel


def styleoptions_widget(frame, side=tk.LEFT, command=None, default_style=None):
    _options_style = options_style if default_style is None else unique([default_style]+options_style)
    return labeled_options(frame, side=side, infobox=False, vartype=tk.StringVar, vardefault=_options_style[0],
                           options=_options_style, command=command)


def coloroptions_widget(frame, side=tk.LEFT, command=None, default_color=None):
    def update_background(*args):
        option_menu.configure(background=to_hex(color_var.get()),
                              foreground='white' if sum([v for v in to_rgb(color_var.get())]) < 0.5*3 else 'black')
    _options_color = (options_color
                      if (default_color is None
                          or to_hex(default_color) in [to_hex(c) for c in options_color])
                      else unique([default_color]+options_color))
    _default = (_options_color[[to_hex(c) for c in options_color].index(to_hex(default_color))]
                if to_hex(default_color) in [to_hex(c) for c in options_color]
                else _options_color[0])
    result = labeled_options(frame, side=side, infobox=False, vartype=tk.StringVar, vardefault=_default,
                             options=_options_color, command=command)
    color_var = result[0]
    color_var.trace_add('write', update_background)
    option_menu = result[3]
    option_menu.configure(background=to_hex(_default),
                          foreground='white' if sum([v for v in to_rgb(_default)]) < 0.5*3 else 'black')
    inner_menu = option_menu.nametowidget(option_menu.cget('menu'))
    for c in _options_color:
        index = inner_menu.index(c)
        inner_menu.entryconfigure(index, background=to_hex(c),
                                  foreground='white' if sum([v for v in to_rgb(c)]) < 0.5*3 else 'black')
    return result


def plot_style_widget(baseframe, text: str = '', command=None, padding=padding_setting, side=tk.LEFT, grid=None,
                      vartype_color=tk.StringVar, vartype_style=tk.StringVar, default_color='k', default_style='-'):
    frame = labeled_widget_frame(baseframe, padding, side, grid)
    labeled_widget_label(frame, text, side=tk.TOP)
    color_out = coloroptions_widget(frame, side=tk.TOP, command=command, default_color=default_color)
    style_out = styleoptions_widget(frame, side=tk.TOP, command=command, default_style=default_style)
    var_color, var_style = color_out[0], style_out[0]
    return var_color, var_style


if __name__ == '__main__':
    from app import run_app  # noqa
    run_app()
