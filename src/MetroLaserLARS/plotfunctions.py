# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:10:40 2021

@author: olson
"""
# External imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.projections as proj
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d.axis3d import Axis
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import ArrayLike
import time
from scipy.stats import gmean
from typing import Literal, Optional


def can_iter(o):
    """
    Checks whether an object `o` is iterable.

    Parameters
    ----------
    o : object
        Object to test.

    Returns
    -------
    bool
        Whether `o` is iterable.

    """
    try:
        _ = iter(o)
    except TypeError:
        return False
    else:
        return True


def color_plot(x=None, y=None, z=None, x_scale=1, y_scale=1, z_scale=1, x_slice=None, y_slice=None,
               extend_to_zero=False, extend_to_zero_y=False, x_label=None, y_label=None, z_label=None, title='',
               show_title=True, show_color_bar=True, color_map='magma', color_map_custom=None, fname=None,
               z_extrema=None, z_extrema_std=None, invert_y=True, v_line_pos=None, v_line_color='black',
               show_contours=True, show_contour_zero=True, contour_color=('k', 'k', 'w'), contour_linestyles=('-', ':'),
               contour_linewidths=(6, 4), show_contour_labels=False, contour_label_format='%1.0f',
               contour_label_spacing=8, auto_contour=False, custom_contours=None, label_all_custom_contours=False,
               contour_color_adjust_factor=1, manual_contour_labels=None, axis_line_width=4, show_plot_in_spyder=True,
               plot_scale=1, font_settings={'weight': 'bold', 'size': 22}, extend_color_bar='both', z_format=None,
               z_modify_exponent=True, z_log=False, z_log_symmetric=False, z_ticks=None, no_tick_labels=False,
               curve_x=None, curve_y=None, curve_color=None, save_folder=None,
               smooth=False, smooth_knot_spacing=(5, 5), smooth_concentrated_knots=0, smooth_knot_location=None,
               smooth_order=(3, 3), smooth_exclude_rows=0, blur=False, blur_sigma=None, blur_exclude_rows=0,
               blur_excluded_sigma=0, rough=None):

    #    color_map = copy.copy(matplotlib.cm.get_cmap(color_map))
    #    color_map.set_bad(color_map.colors[0])
    if x is None or y is None or z is None:
        if x is None and y is None and z is not None:
            x, y = np.meshgrid(np.linspace(0, 1, np.shape(z)[0]), np.linspace(0, 1, np.shape(z)[1]))
        else:
            raise TypeError("""color_plot() missing required arguments: 'x','y', or 'z'.
You must supply 'z' only or all three.""")

    matplotlib.rc('font', **font_settings)
    plot_norm = matplotlib.colors.Normalize()

    if z_format is None and z_modify_exponent:
        z_format = matplotlib.ticker.ScalarFormatter()
        z_format.set_powerlimits((-2, 3))

    if color_map == 'custom':
        if color_map_custom is None:
            color_map = 'magma'
        else:
            from matplotlib.colors import LinearSegmentedColormap
            color_map = LinearSegmentedColormap.from_list('custom', color_map_custom)

    xres = np.shape(z)[0]

    x = x.copy()
    x *= x_scale
    y = y.copy()
    y *= y_scale
    z = z.copy()
    z *= z_scale
    if curve_x is not None:
        curve_x = curve_x.copy()
        curve_x *= x_scale
    if curve_y is not None:
        curve_y = curve_y.copy()
        curve_y *= y_scale


#    if z_log is None:
#        plot_norm = matplotlib.colors.Normalize()
#        plot_norm = matplotlib.colors.LogNorm(vmin=1e-6)
#    else:
#        plot_norm = matplotlib.colors.LogNorm()#(vmin=1e-6,vmax=1,clip=True)

    if z_log_symmetric:
        z = np.sign(z)*np.log10(np.abs(z))
    elif z_log:
        z = np.log10(z)

    if v_line_pos is not None:
        v_line_pos *= x_scale

    if not (y_slice is None or y_slice == (0, 0)) and not (x_slice is None or x_slice == (0, 0)):
        x = x[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
        y = y[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
        z = z[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
        if curve_x is not None:
            curve_x = curve_x[x_slice[0]:x_slice[1]]
        if curve_x is not None:
            curve_y = curve_y[x_slice[0]:x_slice[1]]
    elif not (y_slice is None or y_slice == (0, 0)):
        x = x[y_slice[0]:y_slice[1], :]
        y = y[y_slice[0]:y_slice[1], :]
        z = z[y_slice[0]:y_slice[1], :]
    elif not (x_slice is None or x_slice == (0, 0)):
        x = x[:, x_slice[0]:x_slice[1]]
        y = y[:, x_slice[0]:x_slice[1]]
        z = z[:, x_slice[0]:x_slice[1]]
        if curve_x is not None:
            curve_x = curve_x[x_slice[0]:x_slice[1]]
        if curve_x is not None:
            curve_y = curve_y[x_slice[0]:x_slice[1]]

    if smooth:
        if blur:
            print("Warning: both blur and smooth are on")
        # if smooth_exclude_rows is not None:
        presmoothx = x.copy()
        presmoothy = y.copy()
        presmoothz = z.copy()
        x = x[smooth_exclude_rows:, :]
        y = y[smooth_exclude_rows:, :]
        z = z[smooth_exclude_rows:, :]
        smooth_knot_location = (xres//2, 0) if smooth_knot_location is None else smooth_knot_location
        # dense_knots = 0 # dense knots from [Rres//2-1-dense_knots:Rres//2+dense_knots,0:dense_knots]
        # Rres = np.size(R[0,:])
        # xspacing = smooth_knot_spacing[0]
        # tx = np.concatenate((x[0,0]*np.ones(3),x[0,:smooth_knot_location[0]-1-dense_knots:xspacing],x[0,smooth_knot_location[0]-1-dense_knots:smooth_knot_location[0]+dense_knots-1],x[0,smooth_knot_location[0]+dense_knots:-1:xspacing],x[0,-1]*np.ones(4)))*10**9
        # Zres = np.size(Z[:,0])
        # yspacing = smooth_knot_spacing[1]

        def make_knots(a, ell, d, s):
            if ell == 0:
                return np.concatenate((a[0, 0]*np.ones(3), a[:d, 0], a[d+1:-1:s, 0], a[-1, 0]*np.ones(4)))
            else:
                return np.concatenate((a[0, 0]*np.ones(3), a[0, :ell-1-d:s], a[0, ell-1-d:ell+d-1], a[0, ell+d:-1:s],
                                       a[0, -1]*np.ones(4)))
        tx = make_knots(x, smooth_knot_location[0], smooth_concentrated_knots, smooth_knot_spacing[0])
        ty = make_knots(y, smooth_knot_location[1], smooth_concentrated_knots, smooth_knot_spacing[1])
        # ty = np.concatenate((Z[0,0]*np.ones(3),Z[:dense_knots,0],Z[dense_knots+1:-1:yspacing,0],Z[-1,0]*np.ones(4)))*10**9

        # ======================================================================================
        # Make a bunch of knots right around the contact area, then maybe good?
        # ======================================================================================
        print('starting spline interpolation')
        from scipy.interpolate import bisplrep, bisplev
        newz = np.zeros_like(z)
        time0 = time.time()
        print('bisplrep')
        xscale = gmean(np.abs(x), axis=None)
        yscale = gmean(np.abs(y), axis=None)
        zscale = gmean(np.abs(z), axis=None)
        splfit = bisplrep(x/xscale, y/yscale, z/zscale, kx=smooth_order[0], ky=smooth_order[1], task=-1, tx=tx/xscale,
                          ty=ty/yscale)
        print(time.time()-time0, 'bisplev')
        newz = bisplev(x[0]/xscale, y[:, 0]/yscale, splfit).T*zscale
        print(time.time()-time0, 'done')

        x = presmoothx
        y = presmoothy
        if smooth_exclude_rows > 0:
            z = np.concatenate((presmoothz[:smooth_exclude_rows], newz))
        else:
            z = newz
    if blur:
        preblurz = z.copy()
        z = z[blur_exclude_rows:, :]
        from scipy.ndimage import gaussian_filter
        blur_sigma = 1 if blur_sigma is None else blur_sigma
        # if blur_regions is None:
        z = gaussian_filter(z, sigma=blur_sigma)
        if blur_exclude_rows > 0:
            z = np.concatenate((gaussian_filter(preblurz[:blur_exclude_rows], sigma=blur_excluded_sigma), z))
        # else:
        #     for region in blur_regions:
        #         z[region[0][0]:region[1][0],region[0][1]:region[1][1]] = gaussian_filter(z[region[0][0]:region[1][0],region[0][1]:region[1][1]], sigma=blur_sigma)

    if rough is not None:
        x = x[::rough, ::rough]
        y = y[::rough, ::rough]
        z = z[::rough, ::rough]
    if extend_to_zero:
        # 2nd order finite-difference slope
        if curve_x is not None and curve_y is not None:
            x1, x2, x3, f1, f2, f3 = curve_x[0], curve_x[1], curve_x[2], curve_y[0], curve_y[1], curve_y[2]
            slope = f1*(2*x1-x2-x3)/((x2-x1)*(x3-x1))+f2*(x3-x1)/((x2-x1)*(x3-x2))+f3*(x1-x2)/((x3-x2)*(x3-x1))
            curve_y = np.insert(curve_y, 0, f1-slope*x1)
            curve_x = np.insert(curve_x, 0, 0)
        if invert_y:
            x = np.transpose(np.vstack((np.zeros(np.shape(x)[0]), np.transpose(x))))
            x = np.vstack((x[0, :], x))
            y = np.vstack((np.zeros(np.shape(y)[1]), y))
            y = np.transpose(np.vstack((y[:, 0], np.transpose(y))))
            z = np.vstack((z[0, :], z))
            z = np.transpose(np.vstack((z[:, 0], np.transpose(z))))
        else:
            x = np.transpose(np.vstack((np.zeros(np.shape(x)[0]), np.transpose(x))))
            x = np.vstack((x[0, :], x))
            y = np.vstack((np.zeros(np.shape(y)[1]), y))
            y = np.transpose(np.vstack((y[:, 0], np.transpose(y))))
            z = np.vstack((z, z[-1, :]))
            z = np.transpose(np.vstack((z[:, 0], np.transpose(z))))
    elif extend_to_zero_y:
        if invert_y:
            x = np.vstack((x[0, :], x))
            y = np.vstack((np.zeros(np.shape(y)[1]), y))
            z = np.vstack((z[0, :], z))
        else:
            x = np.vstack((x[0, :], x))
            y = np.vstack((np.zeros(np.shape(y)[1]), y))
            z = np.vstack((z, z[-1, :]))

    if z_extrema_std is not None:
        z_extrema = (np.mean(z)-z_extrema_std*np.std(z), np.mean(z)+z_extrema_std*np.std(z))

    fig = plt.figure()
    if show_color_bar:
        fig.set_size_inches(12*plot_scale, 9*plot_scale)
    else:
        fig.set_size_inches(9*plot_scale, 9*plot_scale)
    ax = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(axis_line_width)
        ax.tick_params(width=axis_line_width, length=2*axis_line_width, direction='out')

    if invert_y:
        ax.invert_yaxis()

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show_title:
        plt.title(title)
    if z_extrema is None:
        im = ax.pcolormesh(x, y, z, cmap=color_map, norm=plot_norm)
    else:
        im = ax.pcolormesh(x, y, z, cmap=color_map, vmin=z_extrema[0], vmax=z_extrema[1])

    if show_contours and not auto_contour:
        css = []
        if z_extrema is None:
            cs = ax.contour(x, y, z, colors=contour_color[0])
            if show_contour_labels:
                ax.clabel(cs, colors=contour_color[0], fmt=contour_label_format)
            css.append(cs)
        elif custom_contours is None:
            level_center = (z_extrema[0]+z_extrema[1])/2
            level_min = z_extrema[0]
            level_max = z_extrema[1]
            level_low = (level_min+level_center)/2
            level_high = (level_max+level_center)/2
            if show_contour_zero:
                cs = ax.contour(x, y, z, [level_center], colors=contour_color[0], linestyles=contour_linestyles[0],
                                linewidths=contour_linewidths[0])
                css.append(cs)
            cs = ax.contour(x, y, z, [level_high, level_max], colors=contour_color[1], linestyles=contour_linestyles[1],
                            linewidths=contour_linewidths[1])
            css.append(cs)
            cs = ax.contour(x, y, z, [level_min, level_low], colors=contour_color[2], linestyles=contour_linestyles[1],
                            linewidths=contour_linewidths[1])
            css.append(cs)

#            cs = ax.contour(x,y,z,[0],colors=contour_color[0],linestyles=contour_linestyles[0],linewidths=contour_linewidths[0])
#            css.append(cs)
#            cs = ax.contour(x,y,z,[z_extrema[1]*1/2,z_extrema[1]*2/2],colors=contour_color[1],linestyles=contour_linestyles[1],linewidths=contour_linewidths[1])\
#                if z_extrema[1]>0 else\
#                ax.contour(x,y,z,[z_extrema[1]*2/2,z_extrema[1]*1/2],colors=contour_color[1],linestyles=contour_linestyles[1],linewidths=contour_linewidths[1])
#            css.append(cs)
#            cs = ax.contour(x,y,z,[z_extrema[0]*1/2,z_extrema[0]*2/2],colors=contour_color[2],linestyles=contour_linestyles[1],linewidths=contour_linewidths[1])\
#                if z_extrema[0]>0 else\
#                ax.contour(x,y,z,[z_extrema[0]*2/2,z_extrema[0]*1/2],colors=contour_color[2],linestyles=contour_linestyles[1],linewidths=contour_linewidths[1])
#            css.append(cs)
        else:
            for ccidx, custom_contour in enumerate(custom_contours):
                if custom_contour > np.min(z) and custom_contour < np.max(z):
                    if custom_contour == 0:
                        if show_contour_zero:
                            if 0 == (z_extrema[0]+z_extrema[1])/2:
                                cs = ax.contour(x, y, z, [0], colors=contour_color[0], linestyles=contour_linestyles[0],
                                                linewidths=contour_linewidths[0])
                            elif 0 >= contour_color_adjust_factor*(z_extrema[0]+z_extrema[1])/2:
                                cs = ax.contour(x, y, z, [0], colors=contour_color[1], linestyles=contour_linestyles[0],
                                                linewidths=contour_linewidths[0])
                            else:
                                cs = ax.contour(x, y, z, [0], colors=contour_color[2], linestyles=contour_linestyles[0],
                                                linewidths=contour_linewidths[0])
                            css.append(cs)
                    elif custom_contour >= contour_color_adjust_factor*(z_extrema[0]+z_extrema[1])/2:
                        cs = ax.contour(x, y, z, [custom_contour], colors=contour_color[1],
                                        linestyles=contour_linestyles[1], linewidths=contour_linewidths[1])
                        if show_contour_labels:
                            if manual_contour_labels is None:
                                ax.clabel(cs, colors=contour_color[1], fmt=contour_label_format,
                                          inline_spacing=contour_label_spacing)
                            else:
                                ax.clabel(cs, colors=contour_color[1], fmt=contour_label_format,
                                          inline_spacing=contour_label_spacing, manual=manual_contour_labels[ccidx])
                        css.append(cs)
                    else:
                        cs = ax.contour(x, y, z, [custom_contour], colors=contour_color[2],
                                        linestyles=contour_linestyles[1], linewidths=contour_linewidths[1])
                        if show_contour_labels:
                            if manual_contour_labels is None:
                                ax.clabel(cs, colors=contour_color[2], fmt=contour_label_format,
                                          inline_spacing=contour_label_spacing)
                            else:
                                ax.clabel(cs, colors=contour_color[2], fmt=contour_label_format,
                                          inline_spacing=contour_label_spacing, manual=manual_contour_labels[ccidx])
                        css.append(cs)
    if show_color_bar and not auto_contour:
        if z_label is None:
            if show_contours and custom_contours is not None:
                cbar = fig.colorbar(im, ax=ax, extend=extend_color_bar, format=z_format, ticks=custom_contours)
            else:
                cbar = fig.colorbar(im, ax=ax, extend=extend_color_bar, format=z_format)
        else:
            if show_contours and custom_contours is not None:
                cbar = fig.colorbar(im, ax=ax, extend=extend_color_bar, label=z_label, format=z_format,
                                    ticks=custom_contours)
            else:
                cbar = fig.colorbar(im, ax=ax, extend=extend_color_bar, label=z_label, format=z_format)
        if z_extrema is None and show_contours:
            cbar.add_lines(cs)
        elif show_contours:
            if custom_contours is None:
                level_center = (z_extrema[0]+z_extrema[1])/2
                level_min = z_extrema[0]
                level_max = z_extrema[1]
                level_low = (level_min+level_center)/2
                level_high = (level_max+level_center)/2
                cbar.ax.plot([-1e9, 1e9], [level_min, level_min], color=contour_color[2],
                             linewidth=contour_linewidths[1], dashes=[1, 1])
                cbar.ax.plot([-1e9, 1e9], [level_low, level_low], color=contour_color[2],
                             linewidth=contour_linewidths[1], dashes=[1, 1])
                cbar.ax.plot([-1e9, 1e9], [level_high, level_high], color=contour_color[1],
                             linewidth=contour_linewidths[1], dashes=[1, 1])
                cbar.ax.plot([-1e9, 1e9], [level_max, level_max], color=contour_color[1],
                             linewidth=contour_linewidths[1], dashes=[1, 1])
                cbar.ax.plot([-1e9, 1e9], [level_center, level_center], color=contour_color[0],
                             linewidth=contour_linewidths[0])
                if show_contour_zero:
                    cbar.ax.hlines(0, -1, 1, colors=contour_color[0],
                                   linewidth=contour_linewidths[0]/2, linestyles=contour_linestyles[0])
            else:
                for custom_contour in custom_contours:
                    if label_all_custom_contours or (custom_contour > np.min(z) and custom_contour < np.max(z)):
                        if custom_contour == 0:
                            if show_contour_zero:
                                if 0 == (z_extrema[0]+z_extrema[1])/2:
                                    cbar.ax.plot([-1e9, 1e9], [0, 0], color=contour_color[0],
                                                 linewidth=contour_linewidths[0])
                                elif 0 >= contour_color_adjust_factor*(z_extrema[0]+z_extrema[1])/2:
                                    cbar.ax.plot([-1e9, 1e9], [0, 0], color=contour_color[1],
                                                 linewidth=contour_linewidths[0])
                                else:
                                    cbar.ax.plot([-1e9, 1e9], [0, 0], color=contour_color[2],
                                                 linewidth=contour_linewidths[0])
                        elif custom_contour >= contour_color_adjust_factor*(z_extrema[0]+z_extrema[1])/2:
                            cbar.ax.plot([-1e9, 1e9], [custom_contour, custom_contour], color=contour_color[1],
                                         linewidth=contour_linewidths[1], dashes=[1, 1])
                        else:
                            cbar.ax.plot([-1e9, 1e9], [custom_contour, custom_contour], color=contour_color[2],
                                         linewidth=contour_linewidths[1], dashes=[1, 1])
    elif show_color_bar:
        if z_label is None:
            cbar = fig.colorbar(im, ax=ax, extend=extend_color_bar, format=z_format)
        else:
            cbar = fig.colorbar(im, ax=ax, extend=extend_color_bar, label=z_label, format=z_format)
        if show_contours:
            cbar.add_lines(cs)

    if show_color_bar and z_ticks is not None:
        cbar.set_ticks(z_ticks)
        cbar.set_ticklabels(z_ticks)

    if v_line_pos is not None:
        if v_line_color == 'kw':
            plt.axvline(x=v_line_pos, color='w', linewidth=12)
            plt.axvline(x=v_line_pos, color='k', linewidth=4)
        elif v_line_color == 'wk':
            plt.axvline(x=v_line_pos, color='k', linewidth=12)
            plt.axvline(x=v_line_pos, color='w', linewidth=4)
        else:
            plt.axvline(x=v_line_pos, color=v_line_color, linewidth=6)

    if curve_x is not None and curve_y is not None:
        if curve_color is None:
            ax.plot(curve_x, curve_y, '-', linewidth=4)
        else:
            ax.plot(curve_x, curve_y, '-'+curve_color, linewidth=4)

    if no_tick_labels:
        ax.set_xticks([])
        ax.set_yticks([])

    # from custommodules.settingshandler import handle_setting
    # _CLUSTERMODE = handle_setting('CLUSTERMODE', False)  # from settings import CLUSTERMODE as _CLUSTERMODE
    _CLUSTERMODE = False
    if fname is not None:
        if save_folder is None:
            if not _CLUSTERMODE:
                print('saving plot to '+fname+'.png ...')
            plt.savefig(fname+'.png', dpi=150, bbox_inches='tight')
        else:
            if not _CLUSTERMODE:
                print('saving plot to', save_folder+'\\'+fname, '.png ...')
            import os
            cwd = os.getcwd()
            os.chdir(save_folder)
            plt.savefig(fname+'.png', dpi=150, bbox_inches='tight')
            os.chdir(cwd)

    if show_plot_in_spyder and not _CLUSTERMODE:
        plt.show()

    plt.close(fig)
    return


def line_plot(x: ArrayLike, y: ArrayLike, legend=None, x_lim: tuple[float, float] = None,
              y_lim: tuple[float, float] = None, fig_size: tuple[float, float] = (12, 9), x_label: str = None,
              y_label: str = None, title: str = None, legend_location='outside', legend_title: str = None,
              fname: str = None, line_width: float = 6, cmap=None, cmap_custom=None, style='-',
              axis_line_width: float = 4, plot_scale: float = 1, show_plot_in_spyder: bool = True,
              save_folder: Optional[str] = None, font_settings: dict = {'weight': 'bold', 'size': 22}, y_format=None,
              x_ticks: Optional[ArrayLike] = None, y_ticks: Optional[ArrayLike] = None,
              v_line_pos: Optional[ArrayLike] = None, v_line_color='k', v_line_width: float = 6,
              v_line_legend=None,
              y_norm: Literal[None, 'global', 'each'] = None, fig=None,
              x_slice: tuple[float, float] = (-np.inf, np.inf), legend_interactive: bool = False,
              clear_fig: bool = True):

    x = x.copy()
    y = y.copy()

    if isinstance(x, list) and isinstance(y, list):
        for i, (xel, yel) in enumerate(zip(x, y)):
            idxs = (np.where(xel >= x_slice[0])[0][0] if np.where(xel >= x_slice[0])[0].size > 0 else 0,
                    np.where(xel > x_slice[1])[0][0] if np.where(xel > x_slice[1])[0].size > 0 else len(xel))
            x[i] = xel[slice(*idxs)]
            y[i] = yel[slice(*idxs)]
    else:
        idxs = (np.where(x >= x_slice[0])[0][0] if np.where(x >= x_slice[0])[0].size > 0 else 0,
                np.where(x > x_slice[1])[0][0] if np.where(x > x_slice[1])[0].size > 0 else len(x))
        x = x[slice(*idxs)]
        if isinstance(y, list):
            for i, yel in enumerate(y):
                y[i] = yel[slice(*idxs)]
        else:
            y = y[slice(*idxs)]

    if y_norm is not None:
        if y_norm == 'global':
            globalmax = np.nanmax(np.abs(y))
            if any(isinstance(el, list) for el in y) or isinstance(y, list):
                for idx, tmpy in enumerate(y):
                    y[idx] = tmpy/globalmax
            else:
                y /= globalmax
        elif y_norm == 'each':
            if any(isinstance(el, list) for el in y) or isinstance(y, list):
                for idx, tmpy in enumerate(y):
                    y[idx] = tmpy/np.nanmax(np.abs(tmpy))
            else:
                y /= np.nanmax(np.abs(y))

    matplotlib.rc('font', **font_settings)

    if fig is None:
        using_extant_figure = False
        fig = plt.figure()
        fig.set_size_inches(fig_size[0], fig_size[1])
    else:
        using_extant_figure = True
        if clear_fig:
            fig.clear()
    ax = fig.gca()
    extantlinelist = []
    if using_extant_figure:
        for leg in fig.legends:
            leg.remove()
        for axis in fig.axes:
            for line in axis.lines:
                extantlinelist.append(line)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(axis_line_width)
        ax.tick_params(width=axis_line_width, length=2*axis_line_width, direction='in')

    if cmap is not None:
        if cmap == 'list' and cmap_custom is not None:
            colors = []
            for i in range(len(y)):
                colors.append(cmap_custom[i])
        else:
            if cmap == 'custom':
                if cmap_custom is None:
                    cmap = 'magma'
                else:
                    from matplotlib.colors import LinearSegmentedColormap
                    cmap = LinearSegmentedColormap.from_list('custom', cmap_custom)
            colormap = matplotlib.cm.get_cmap(cmap)
            colors = []
            for i in range(len(y)):
                colors.append(colormap((i+1)/(len(y)+1)))

    if any(isinstance(el, list) for el in y) or isinstance(y, list):
        if any(isinstance(el, list) for el in x) or isinstance(x, list):
            for idx, (plotx, ploty) in enumerate(zip(x, y)):
                if cmap is not None:
                    ax.plot(plotx, ploty, style, color=colors[idx],
                            label=legend[idx] if legend is not None else '')
                else:
                    ax.plot(plotx, ploty, style,
                            label=legend[idx] if legend is not None else '')
        else:
            for idx, ploty in enumerate(y):
                if cmap is not None:
                    ax.plot(x, ploty, style, color=colors[idx],
                            label=legend[idx] if legend is not None else '')
                else:
                    ax.plot(x, ploty, style,
                            label=legend[idx] if legend is not None else '')
    else:
        ax.plot(x, y, style,
                label=legend if legend is not None else '')

    if y_format is not None:
        ax.yaxis.set_major_formatter(y_format)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    linelist = []
    for axis in fig.axes:
        for line in axis.lines:
            if line not in extantlinelist:
                linelist.append(line)

    v_line_legend_used = [False for vll in v_line_legend] if v_line_legend is not None else None
    if v_line_pos is not None:
        if can_iter(v_line_pos):  # multiple vlines?
            if can_iter(v_line_pos[0]):  # multiple sets of vlines?
                if can_iter(v_line_color) and not isinstance(v_line_color, str):  # multiple colors?
                    for vlgroup, (vlps, vlc, vlw) in enumerate(zip(v_line_pos, v_line_color, v_line_width)):
                        for vlp in vlps:
                            ax.axvline(x=vlp, color=vlc, linewidth=vlw,
                                       label=(v_line_legend[vlgroup] if
                                              (v_line_legend is not None and not v_line_legend_used[vlgroup])
                                              else (
                                                  '_'+v_line_legend[vlgroup] if
                                                  (v_line_legend is not None)
                                                  else '')
                                              )
                                       )
                            if v_line_legend is not None:
                                v_line_legend_used[vlgroup] = True
                else:  # one color
                    for vlps in v_line_pos:
                        for vlp in vlps:
                            ax.axvline(x=vlp, color=v_line_color, linewidth=v_line_width,
                                       label=(v_line_legend[0] if
                                              (v_line_legend is not None and not v_line_legend_used[0])
                                              else (
                                                  '_'+v_line_legend[0] if
                                                  (v_line_legend is not None)
                                                  else '')
                                              )
                                       )
                            if v_line_legend is not None:
                                v_line_legend_used[0] = True
            else:  # one set of vlines
                if can_iter(v_line_color) and not isinstance(v_line_color, str):  # multiple colors?
                    for vlgroup, (vlp, vlc) in enumerate(zip(v_line_pos, v_line_color)):
                        ax.axvline(x=vlp, color=vlc, linewidth=v_line_width,
                                   label=(v_line_legend[vlgroup] if
                                          (v_line_legend is not None and not v_line_legend_used[vlgroup])
                                          else (
                                              '_'+v_line_legend[vlgroup] if
                                              (v_line_legend is not None)
                                              else '')
                                          )
                                   )
                        if v_line_legend is not None:
                            v_line_legend_used[vlgroup] = True
                else:  # one color
                    for vlp in v_line_pos:
                        ax.axvline(x=vlp, color=v_line_color, linewidth=v_line_width,
                                   label=(v_line_legend[0] if
                                          (v_line_legend is not None and not v_line_legend_used[0])
                                          else (
                                              '_'+v_line_legend[0] if
                                              (v_line_legend is not None)
                                              else '')
                                          )
                                   )
                        if v_line_legend is not None:
                            v_line_legend_used[0] = True
        else:  # one vline
            ax.axvline(x=v_line_pos, color=v_line_color, linewidth=v_line_width,
                       label=(v_line_legend[0] if
                              (v_line_legend is not None and not v_line_legend_used[0])
                              else (
                                  '_'+v_line_legend[0] if
                                  (v_line_legend is not None)
                                  else '')
                              )
                       )
            if v_line_legend is not None:
                v_line_legend_used[0] = True

    vlinelist = []
    for axis in fig.axes:
        for line in axis.lines:
            if line not in extantlinelist and line not in linelist:
                vlinelist.append(line)

    for line in linelist:
        line.set_linewidth(line_width)

    if legend is not None:
        leglinelist = extantlinelist+linelist+vlinelist if v_line_legend is not None else extantlinelist+linelist
        legend_plot_source = fig if using_extant_figure else plt
        if legend_location == 'outside':
            leg = legend_plot_source.legend(handles=leglinelist, title=legend_title,
                                            bbox_to_anchor=(1.04, 1), loc='upper left',
                                            framealpha=1, fancybox=False)
        else:
            leg = legend_plot_source.legend(handles=leglinelist, loc=legend_location,
                                            title=legend_title, framealpha=1, fancybox=False)
        leg.get_frame().set_linewidth(axis_line_width)
        leg.get_frame().set_edgecolor('black')

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if title is not None:
        ax.set_title(title)

    if legend_interactive:
        map_legend_to_ax = {}
        pickradius = int(font_settings['size']/2)

        # for legend_line, ax_line in zip(leg.get_lines(), leglinelist):
        #     legend_line.set_picker(pickradius)
        #     map_legend_to_ax[legend_line] = ax_line

        for legend_line in leg.get_lines():
            map_legend_to_ax[legend_line] = []
            for ax_line in leglinelist:
                if ((ax_line.get_label() == legend_line.get_label())
                        or (ax_line.get_label()[0] == '_' and ax_line.get_label()[1:] == legend_line.get_label())):
                    map_legend_to_ax[legend_line].append(ax_line)

        for h in leg.legend_handles:
            h.set_picker(pickradius)

        def on_pick(event):
            legend_line = event.artist

            if legend_line not in map_legend_to_ax:
                return

            ax_lines = map_legend_to_ax[legend_line]
            for ax_line in ax_lines:
                visible = not ax_line.get_visible()
                ax_line.set_visible(visible)
            legend_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)
        leg.set_draggable(True)

    plt.draw()

    # try:
    #     from settings import CLUSTERMODE as _CLUSTERMODE
    # except ModuleNotFoundError:
    _CLUSTERMODE = True
    if fname is not None:
        if save_folder is None:
            if not _CLUSTERMODE:
                print('saving plot to '+fname+'.png ...')
            plt.savefig(fname+'.png', dpi=150, bbox_inches='tight')
        else:
            if not _CLUSTERMODE:
                print('saving plot to', save_folder+'\\'+fname, '.png ...')
            import os
            cwd = os.getcwd()
            os.chdir(save_folder)
            plt.savefig(fname+'.png', dpi=150, bbox_inches='tight')
            os.chdir(cwd)

    if show_plot_in_spyder:
        plt.show()

    plt.close(fig)
    return fig


def radial_plot(x, y, legend=None, x_lim=None, y_lim=None, fig_size=(12, 9),
                x_label=None, y_label=None, title=None, legend_location=(1.1, 0.4), legend_title=None,
                fname=None, line_width=6, cmap=None, show_plot_in_spyder=True,
                axis_line_width=4, plot_scale=1, fix_x_order=True, r_lim=None,
                r_ticks=None, font_settings={'weight': 'bold', 'size': 22}, frame='circle', save_folder=None):
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D

    def radar_factory(num_vars, frame='circle'):
        """
        Create a radar chart with `num_vars` axes.

        This function creates a RadarAxes projection and registers it.

        Parameters
        ----------
        num_vars : int
            Number of variables for radar chart.
        frame : {'circle', 'polygon'}
            Shape of frame surrounding axes.

        """
        # calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarTransform(PolarAxes.PolarTransform):

            def transform_path_non_affine(self, path):
                # Paths with non-unit interpolation steps correspond to gridlines,
                # in which case we force interpolation (to defeat PolarTransform's
                # autoconversion to circular arcs).
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

        class RadarAxes(PolarAxes):

            name = 'radar'
            # use 1 line segment to connect specified points
            RESOLUTION = 1
            PolarTransform = RadarTransform

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that the first axis is at the top
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that line is closed by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that line is closed by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                # FIXME: markers at x[0], y[0] get doubled-up
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                # in axes coordinates.
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars,
                                          radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                    spine = Spine(axes=self,
                                  spine_type='circle',
                                  path=Path.unit_regular_polygon(num_vars))
                    # unit_regular_polygon gives a polygon of radius 1 centered at
                    # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                    # 0.5) in axes coordinates.
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                        + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta

    matplotlib.rc('font', **font_settings)

    N = len(x)
    theta = radar_factory(N, frame=frame)
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection='radar'))

    x_label = [str(el)+u'\N{DEGREE SIGN}' for el in x] if x_label is None else x_label
    if len(x_label) > 18:
        x_label[1::2] = ['']*len(x_label[1::2])
    if fix_x_order:
        x_label = np.roll(np.flip(x_label), 1)
        x = np.roll(np.flip(x), 1)
        if y.ndim == 1:
            y = np.roll(np.flip(y), 1)
        else:
            for idx, ploty in enumerate(y):
                y[idx] = np.roll(np.flip(ploty), 1)

    if cmap is not None:
        colormap = matplotlib.cm.get_cmap(cmap)
        colors = []
        for i in range(len(y)):
            colors.append(colormap((i+1)/(len(y)+1)))

    if r_lim is None:
        def custom_round(n, d=0, direction=1):
            d = 0 if d < 0 else d
            if direction == -1:
                if d == 0:
                    return np.floor(n)
                else:
                    f = 10**d
                    return np.floor(n*f)/f
            else:
                if d == 0:
                    return np.ceil(n)
                else:
                    f = 10**d
                    return np.ceil(n*f)/f
        if (np.max(y)-np.min(y))/np.mean(np.abs(y)) < 1e-9:
            r_lim_min = np.min(y)*0.9 if np.min(y) > 0 else np.min(y)*1.1
            r_lim_max = np.max(y)*1.1 if np.max(y) > 0 else np.max(y)*0.9
            r_lim_min = custom_round(r_lim_min, round(1-np.log10(r_lim_max-r_lim_min)), -1)
            r_lim_max = custom_round(r_lim_max, round(1-np.log10(r_lim_max-r_lim_min)))
        else:
            r_lim_min = custom_round(np.min(y), round(1-np.log10(np.max(y)-np.min(y))), -1)
            r_lim_max = custom_round(np.max(y), round(1-np.log10(np.max(y)-np.min(y))))
        r_lim = (r_lim_min, r_lim_max)

    if r_ticks is not None:
        ax.set_rgrids(r_ticks)
    else:
        r_grid_spacing = round((r_lim[1]-r_lim[0])/5, round(1-np.log10(r_lim[1]-r_lim[0])))
        r_grid_list = [r_lim[0]]
        i = 0
        while max(r_grid_list) < r_lim[1]:
            i += 1
            r_grid_list.append(r_lim[0]+r_grid_spacing*i)
        ax.set_rgrids(r_grid_list)
    # if title is not None:
    #     ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
    ax.set_varlabels(x_label)

    if y.ndim == 1:
        if cmap is not None:
            ax.plot(theta, y, color=colors[0])
            ax.fill(theta, y, facecolor=colors[0], alpha=0.25, label='_nolegend_')
        else:
            ax.plot(theta, y)
            ax.fill(theta, y, alpha=0.25, label='_nolegend_')
    else:
        for idx, ploty in enumerate(y):
            if cmap is not None:
                ax.plot(theta, ploty, color=colors[idx])
                ax.fill(theta, ploty, facecolor=colors[idx], alpha=0.25, label='_nolegend_')
            else:
                ax.plot(theta, ploty)
                ax.fill(theta, ploty, alpha=0.25, label='_nolegend_')

    if legend is not None:
        ax.legend(legend, loc=legend_location)

    if title is not None:
        plt.title(title)

    plt.draw()

    try:
        from settings import CLUSTERMODE as _CLUSTERMODE
    except ModuleNotFoundError:
        _CLUSTERMODE = True

    if fname is not None:
        if save_folder is None:
            if not _CLUSTERMODE:
                print('saving plot to '+fname+'.png ...')
            plt.savefig(fname+'.png', dpi=150, bbox_inches='tight')
        else:
            if not _CLUSTERMODE:
                print('saving plot to', save_folder+'\\'+fname, '.png ...')
            import os
            cwd = os.getcwd()
            os.chdir(save_folder)
            plt.savefig(fname+'.png', dpi=150, bbox_inches='tight')
            os.chdir(cwd)

    if show_plot_in_spyder and not _CLUSTERMODE:
        plt.show()

    plt.close(fig)
    return True


class axis3d_custom(Axis):
    def __init__(self, adir, v_intervalx, d_intervalx, axes, *args, **kwargs):
        Axis.__init__(self, adir, v_intervalx, d_intervalx, axes, *args, **kwargs)
        self.gridline_colors = []

    def set_gridline_color(self, *gridline_info):
        '''Gridline_info is a tuple containing the value of the gridline to change
        and the color to change it to. A list of tuples may be used with the * operator.'''
        self.gridline_colors.extend(gridline_info)

    def draw(self, renderer):
        # filter locations here so that no extra grid lines are drawn
        Axis.draw(self, renderer)
        which_gridlines = []
        if self.gridline_colors:
            locmin, locmax = self.get_view_interval()
            if locmin > locmax:
                locmin, locmax = locmax, locmin

            # Rudimentary clipping
            majorLocs = [loc for loc in self.major.locator() if
                         locmin <= loc <= locmax]
            for i, val in enumerate(majorLocs):
                for colored_val, color in self.gridline_colors:
                    if val == colored_val:
                        which_gridlines.append((i, color))
            colors = self.gridlines.get_colors()
            for val, color in which_gridlines:
                colors[val] = colorConverter.to_rgba(color)
            self.gridlines.set_color(colors)
            self.gridlines.draw(renderer, project=True)


class XAxis(axis3d_custom):
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.xy_dataLim.intervalx


class YAxis(axis3d_custom):
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.xy_dataLim.intervaly


class ZAxis(axis3d_custom):
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.zz_dataLim.intervalx


class Axes3D_custom(Axes3D):
    """
    3D axes object.
    """
    name = '3d_custom'

    def _init_axis(self):
        '''Init 3D axes; overrides creation of regular X/Y axes'''
        self.w_xaxis = XAxis('x', self.xy_viewLim.intervalx,
                             self.xy_dataLim.intervalx, self)
        self.xaxis = self.w_xaxis
        self.w_yaxis = YAxis('y', self.xy_viewLim.intervaly,
                             self.xy_dataLim.intervaly, self)
        self.yaxis = self.w_yaxis
        self.w_zaxis = ZAxis('z', self.zz_viewLim.intervalx,
                             self.zz_dataLim.intervalx, self)
        self.zaxis = self.w_zaxis

        for ax in self.xaxis, self.yaxis, self.zaxis:
            ax.init3d()


proj.projection_registry.register(Axes3D_custom)


def cylinder_plot(R, Z, T, c, tlims=(0, 180), rlims=(0, 1), zlims=(0, 1), clims=(-2, 2),
                  interpolation_amount=1, dpi=150, figsize=(13, 13), markersize=1, fname=None, save_folder=None,
                  show_plot_in_spyder=True, rticks=None, rlabel=None, zlabel=None, clabel=None, fontsize_axes=16,
                  fontsize_ticks=12, show_color_bar=True, extend_color_bar='both', c_format=None,
                  font_settings={'weight': 'bold', 'size': 24}, c_modify_exponent=True, c_map="magma",
                  c_map_custom=None, r_interpolation=None, z_interpolation=None, c_ticks=None):

    if z_interpolation is not None:
        zint = z_interpolation
        newZ = np.zeros(((np.shape(Z)[0]-1)*zint+1, np.shape(Z)[1]))
        newc = np.zeros(((np.shape(c)[0]-1)*zint+1,)+np.shape(c)[1:])
        newR = np.stack([R[0]]*((np.shape(R)[0]-1)*zint+1))
        for i in range(np.shape(Z)[0]-1):
            newZ[i*zint] = Z[i]
            newc[i*zint] = c[i]
            for k in range(zint):
                newZ[i*zint+k+1] = Z[i]*((zint-k-1)/zint)+Z[i+1]*((k+1)/zint)
                newc[i*zint+k+1] = c[i]*((zint-k-1)/zint)+c[i+1]*((k+1)/zint)
        newZ[-1] = Z[-1]
        newc[-1] = c[-1]

        c = newc
        R = newR
        Z = newZ

    if r_interpolation is not None:
        rint = r_interpolation
        newR = np.zeros((np.shape(Z)[0], (np.shape(R)[1]-1)*rint+1))
        newc = np.zeros((np.shape(c)[0], (np.shape(c)[1]-1)*rint+1, np.shape(c)[2]))
        newZ = np.stack([Z[:, 0]]*((np.shape(Z)[1]-1)*rint+1), axis=1)
        for j in range(np.shape(R)[1]-1):
            newR[:, j*rint] = R[:, j]
            newc[:, j*rint] = c[:, j]
            for k in range(rint):
                newR[:, j*rint+k+1] = R[:, j]*((rint-k-1)/rint)+R[:, j+1]*((k+1)/rint)
                newc[:, j*rint+k+1] = c[:, j]*((rint-k-1)/rint)+c[:, j+1]*((k+1)/rint)
        newR[:, -1] = R[:, -1]
        newc[:, -1] = c[:, -1]

        c = newc
        R = newR
        Z = newZ

    matplotlib.rc('font', **font_settings)

    if c_format is None and c_modify_exponent:
        c_format = matplotlib.ticker.ScalarFormatter()
        c_format.set_powerlimits((-2, 3))

    if c_map == "custom":
        if c_map_custom is None:
            c_map = "magma"
        else:
            from matplotlib.colors import LinearSegmentedColormap
            c_map = LinearSegmentedColormap.from_list('custom', c_map_custom)

    lowangle = tlims[0]
    highangle = tlims[1]
    lowr = rlims[0]
    highr = rlims[1]
    lowz = zlims[0]
    highz = zlims[1]
    lowidxt = np.where(T[:] >= lowangle)[0][0]
    highidxt = np.where(T[:] >= highangle)[0][0]
    lowidxr = np.where(R[0, :] >= lowr)[0][0]
    highidxr = np.where(R[0, :] >= highr)[0][0]
    lowidxz = np.where(Z[:, 0] >= lowz)[0][0]+1
    highidxz = np.where(Z[:, 0] >= highz)[0][0]
    highr_real = R[0, highidxr]
    lowz_real = Z[lowidxz, 0]
    highz_real = Z[highidxz, 0]
    plotx = np.array([])
    ploty = np.array([])
    plotz = np.array([])
    plotc = np.array([])

    black_value = -1e20

    # all angles
    # first many angles
    for i in range(lowidxt, highidxt, 1):
        for interpolate in np.linspace(0, 1, interpolation_amount, endpoint=False):
            # ridxlow = lowidxr+1 if interpolate == 0 and (i == lowidxt or i == highidxt) else lowidxr+1
            # ridxhigh = highidxr if interpolate == 0 and (i == lowidxt or i == highidxt) else highidxr
            # zidxlow = lowidxz+1 if interpolate == 0 and (i == lowidxt or i == highidxt) else lowidxz
            # zidxhigh = highidxz if interpolate == 0 and (i == lowidxt or i == highidxt) else highidxz+1

            # tmpplotc = (1-interpolate)*np.ravel(c[zidxlow:zidxhigh,ridxlow:ridxhigh,i])+interpolate*np.ravel(c[zidxlow:zidxhigh,ridxlow:ridxhigh,i+1])
            # tmpplotz = np.ravel(Z[zidxlow:zidxhigh,ridxlow:ridxhigh])
            # tmpplotr = np.ravel(R[zidxlow:zidxhigh,ridxlow:ridxhigh])
            # tmpplott = (1-interpolate)*T[i]+interpolate*T[i+1]
            # tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
            # tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
            # plotx = np.append(plotx,tmpplotx)
            # ploty = np.append(ploty,tmpploty)
            # plotz = np.append(plotz,tmpplotz)
            # plotc = np.append(plotc,tmpplotc)

            # outside and inside
            for ridx in [lowidxr, highidxr]:
                # don't plot for high/low angle edges
                if not (interpolate == 0 and (i == lowidxt or i == highidxt)):
                    tmpplotc = (1-interpolate)*np.ravel(c[lowidxz+1:highidxz, ridx, i])\
                        + interpolate*np.ravel(c[lowidxz+1:highidxz, ridx, i+1])
                    tmpplotz = np.ravel(Z[lowidxz+1:highidxz, ridx])
                    tmpplotr = np.ravel(R[lowidxz+1:highidxz, ridx])
                    tmpplott = (1-interpolate)*T[i]+interpolate*T[i+1]
                    tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
                    tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
                    plotx = np.append(plotx, tmpplotx)
                    ploty = np.append(ploty, tmpploty)
                    plotz = np.append(plotz, tmpplotz)
                    plotc = np.append(plotc, tmpplotc)

            # top and bottom
            for zidx in [lowidxz, highidxz]:
                # don't plot for high/low angle edges
                if not (interpolate == 0 and (i == lowidxt or i == highidxt)):
                    tmpplotc = (1-interpolate)*np.ravel(c[zidx, lowidxr+1:highidxr, i])\
                        + interpolate*np.ravel(c[zidx, lowidxr+1:highidxr, i+1])
                    tmpplotz = np.ravel(Z[zidx, lowidxr+1:highidxr])
                    tmpplotr = np.ravel(R[zidx, lowidxr+1:highidxr])
                    tmpplott = (1-interpolate)*T[i]+interpolate*T[i+1]
                    tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
                    tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
                    plotx = np.append(plotx, tmpplotx)
                    ploty = np.append(ploty, tmpploty)
                    plotz = np.append(plotz, tmpplotz)
                    plotc = np.append(plotc, tmpplotc)

    # first and last angle
    for tidx in [lowidxt, highidxt]:
        tmpplotc = np.ravel(c[lowidxz+1:highidxz, lowidxr+1:highidxr, tidx])
        tmpplotz = np.ravel(Z[lowidxz+1:highidxz, lowidxr+1:highidxr])
        tmpplotr = np.ravel(R[lowidxz+1:highidxz, lowidxr+1:highidxr])
        tmpplott = T[tidx]
        tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
        tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
        plotx = np.append(plotx, tmpplotx)
        ploty = np.append(ploty, tmpploty)
        plotz = np.append(plotz, tmpplotz)
        plotc = np.append(plotc, tmpplotc)

    # fill in curved faces
    for i in range(lowidxt, highidxt, 1):
        for interpolate in np.linspace(0, 1, interpolation_amount, endpoint=False):
            ridxlist = [lowidxr+1, highidxr-1] if interpolate == 0 and (i == lowidxt or i == highidxt)\
                else [lowidxr, highidxr]
            for ridx in ridxlist:
                zidxlow = lowidxz+1
                zidxhigh = highidxz
                tmpplotc = (1-interpolate)*np.ravel(c[zidxlow:zidxhigh, ridx, i])\
                    + interpolate*np.ravel(c[zidxlow:zidxhigh, ridx, i+1])
                tmpplotz = np.ravel(Z[zidxlow:zidxhigh, ridx])
                tmpplotr = np.ravel(R[zidxlow:zidxhigh, ridx])
                tmpplott = (1-interpolate)*T[i]+interpolate*T[i+1]
                tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
                tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
                plotx = np.append(plotx, tmpplotx)
                ploty = np.append(ploty, tmpploty)
                plotz = np.append(plotz, tmpplotz)
                plotc = np.append(plotc, tmpplotc)

    # black lines at curved edges
    for i in range(lowidxt, highidxt, 1):
        for interpolate in np.linspace(0, 1, interpolation_amount, endpoint=False):
            for zidx in [lowidxz, highidxz]:
                for ridx in [lowidxr, highidxr]:
                    tmpplotc = black_value*np.ravel(np.ones_like(c[zidx, ridx, i]))
                    tmpplotz = np.ravel(Z[zidx, ridx])
                    tmpplotr = np.ravel(R[zidx, ridx])
                    tmpplott = (1-interpolate)*T[i]+interpolate*T[i+1]
                    tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
                    tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
                    plotx = np.append(plotx, tmpplotx)
                    ploty = np.append(ploty, tmpploty)
                    plotz = np.append(plotz, tmpplotz)
                    plotc = np.append(plotc, tmpplotc)

    # black lines at straight edges
    for i in [lowidxt, highidxt]:
        # left and right lines
        for ridx in [lowidxr, highidxr]:
            tmpplotc = black_value*np.ravel(np.ones_like(c[lowidxz:highidxz, ridx, i]))
            tmpplotz = np.ravel(Z[lowidxz:highidxz, ridx])
            tmpplotr = np.ravel(R[lowidxz:highidxz, ridx])
            tmpplott = T[i]
            tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
            tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
            plotx = np.append(plotx, tmpplotx)
            ploty = np.append(ploty, tmpploty)
            plotz = np.append(plotz, tmpplotz)
            plotc = np.append(plotc, tmpplotc)

        # top and bottom lines
        for zidx in [lowidxz, highidxz]:
            tmpplotc = black_value*np.ravel(np.ones_like(c[zidx, lowidxr:highidxr, i]))
            tmpplotz = np.ravel(Z[zidx, lowidxr:highidxr])
            tmpplotr = np.ravel(R[zidx, lowidxr:highidxr])
            tmpplott = T[i]
            tmpplotx = tmpplotr*np.cos(tmpplott*np.pi/180)
            tmpploty = tmpplotr*np.sin(tmpplott*np.pi/180)
            plotx = np.append(plotx, tmpplotx)
            ploty = np.append(ploty, tmpploty)
            plotz = np.append(plotz, tmpplotz)
            plotc = np.append(plotc, tmpplotc)

    # rticks
    num_rticks = 4
    if rticks is None:
        rticks = [lowr*(1-i/num_rticks)+highr*i/num_rticks for i in range(num_rticks+1)]
        rticks = [round(rt, 1) for rt in rticks]
    for rt in rticks:
        rtickdensity = 300
        rt_t = np.linspace(0, 2*np.pi, round(rtickdensity*rt/highr), endpoint=False)
        plotx = np.append(plotx, rt*np.cos(rt_t))
        ploty = np.append(ploty, rt*np.sin(rt_t))
        plotz = np.append(plotz, highz_real*np.ones(round(rtickdensity*rt/highr)))
        plotc = np.append(plotc, black_value*np.ones(round(rtickdensity*rt/highr)))

    # bounding box
    bound_res = 100

    highr_box = highr_real*1.02
    highz_box = highz_real*1.02
    lowz_box = lowz_real-(highz_real-lowz_real)/50
    # #bottom box x
    for bound_y in [-highr_box, highr_box]:
        plotx = np.append(plotx, np.linspace(-highr_box, highr_box, bound_res))
        ploty = np.append(ploty, bound_y*np.ones(bound_res))
        plotz = np.append(plotz, highz_box*np.ones(bound_res))
        plotc = np.append(plotc, black_value*np.ones(bound_res))
    # bottom box y
    for bound_x in [-highr_box, highr_box]:
        plotx = np.append(plotx, bound_x*np.ones(bound_res))
        ploty = np.append(ploty, np.linspace(-highr_box, highr_box, bound_res))
        plotz = np.append(plotz, highz_box*np.ones(bound_res))
        plotc = np.append(plotc, black_value*np.ones(bound_res))
    # verticals
    for (x, y) in [(-highr_box, -highr_box), (-highr_box, highr_box), (highr_box, highr_box)]:
        plotx = np.append(plotx, x*np.ones(bound_res))
        ploty = np.append(ploty, y*np.ones(bound_res))
        plotz = np.append(plotz, np.linspace(lowz_box, highz_box, bound_res))
        plotc = np.append(plotc, black_value*np.ones(bound_res))
    # top x
    plotx = np.append(plotx, np.linspace(-highr_box, highr_box, bound_res))
    ploty = np.append(ploty, highr_box*np.ones(bound_res))
    plotz = np.append(plotz, lowz_box*np.ones(bound_res))
    plotc = np.append(plotc, black_value*np.ones(bound_res))
    # top y
    plotx = np.append(plotx, -highr_box*np.ones(bound_res))
    ploty = np.append(ploty, np.linspace(-highr_box, highr_box, bound_res))
    plotz = np.append(plotz, lowz_box*np.ones(bound_res))
    plotc = np.append(plotc, black_value*np.ones(bound_res))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    norm = matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1], clip=False)
    scatter = ax.scatter(plotx, ploty, plotz, c=plotc, s=matplotlib.rcParams['lines.markersize']**2*markersize**2,
                         edgecolor="none", alpha=1, marker=",", cmap=c_map, linewidth=0, norm=norm, plotnonfinite=True)

    ax.set_xlim((-highr, highr))
    ax.set_ylim((-highr, highr))
    ax.set_zlim((lowz, highz))

    ax.invert_zaxis()

    # ax.yaxis.set_tick_params(which='both',length=0)
    ax.set_xticks(rticks+[-rt for rt in rticks])
    ax.set_yticks([])
    ax.set_yticks(rticks+[-rt for rt in rticks], minor=True)
    ax.set_xticklabels([abs(rt) for rt in rticks+[-rt for rt in rticks]], rotation=58)
    ax.set_yticklabels(['' for rt in rticks+[-rt for rt in rticks]], minor=True)
    ax.xaxis.set_tick_params(pad=-2*figsize[0]/6.5)
    ax.zaxis.set_tick_params(pad=6*figsize[0]/6.5)
    # ax.zaxis.set_tick_params(labelsize=fontsize_ticks)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['tick']['outward_factor'] = 0.0
        axis._axinfo['tick']['inward_factor'] = 0.0

    # for label in ax.get_xticklabels():
    #     label.set_fontweight(0)
    # for label in ax.get_zticklabels():
    #     label.set_fontweight(0)

    if rlabel is not None:
        ax.set_xlabel(rlabel, labelpad=12*figsize[0]/6.5)  # ),fontsize=fontsize_axes,fontweight='bold')
    if zlabel is not None:
        ax.set_zlabel(zlabel, labelpad=14*figsize[0]/6.5)  # ),fontsize=fontsize_axes,fontweight='bold')

    if show_color_bar:
        if clabel is None:
            cbar = fig.colorbar(scatter, ax=ax, extend=extend_color_bar, shrink=0.55, pad=0.11)
        else:
            cbar = fig.colorbar(scatter, ax=ax, extend=extend_color_bar, label=clabel, format=c_format, shrink=0.55,
                                pad=0.11)
    if show_color_bar and c_ticks is not None:
        cbar.set_ticks(c_ticks)
        # cbar.set_ticklabels(c_ticks)
    plt.tight_layout()
    from settings import CLUSTERMODE as _CLUSTERMODE
    if fname is not None:
        if save_folder is None:
            if not _CLUSTERMODE:
                print('saving plot to '+fname+'.png ...')
            plt.savefig(fname+'.png', dpi=dpi, bbox_inches='tight')
        else:
            if not _CLUSTERMODE:
                print('saving plot to', save_folder+'\\'+fname, '.png ...')
            import os
            cwd = os.getcwd()
            os.chdir(save_folder)
            plt.savefig(fname+'.png', dpi=dpi, bbox_inches='tight')
            os.chdir(cwd)

    if show_plot_in_spyder and not _CLUSTERMODE:
        plt.show()

    plt.close(fig)
