# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:57:17 2025

@author: KOlson
"""

import tkinter as tk
# from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import hdbscan
from sklearn.manifold import MDS

try:
    from plotfunctions import line_plot
    from app_helpers import make_window
except ModuleNotFoundError:
    from MetroLaserLARS.plotfunctions import line_plot  # type: ignore
    from MetroLaserLARS.app_helpers import make_window  # type: ignore


def open_clustering_window(root, pair_results_var):

    window = make_window(root, 'Clustering', (800, 900))

    pair_results = pair_results_var.get()

    names = []
    for pr in pair_results:
        names.append(pr['names'][0])
        names.append(pr['names'][1])
    names = sorted(list(set(names)))

    names = [n for n in names if '_' not in n and int(n[-1]) % 2 != 0]  # and n[:2] == '61'

    num_points = len(names)

    # maxq = max([0]+[pr['quality'] for pr in pair_results])

    pairwise_distances = np.zeros((num_points, num_points))
    for pr in pair_results:
        n0 = pr['names'][0]
        n1 = pr['names'][1]
        if n0 not in names or n1 not in names:
            continue
        i0 = names.index(n0)
        i1 = names.index(n1)
        # pairwise_distances[i0, i1] = pairwise_distances[i1, i0] = 1-pr['match_probability']  # + pr['quality']/maxq
        # pairwise_distances[i0, i1] = pairwise_distances[i1, i0] = (10_000**np.abs(1-pr['stretch'])*(1 - min(0.75, pr['match_probability'])))**3
        pairwise_distances[i0, i1] = pairwise_distances[i1, i0] = (10_000**np.abs(1-pr['stretch']))**3

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='precomputed')
    clusterer.fit(pairwise_distances)
    print(f'{"Name": >10}: {"Cluster"}')

    for i in range(num_points):
        print(f'{names[i]: >10}: {clusterer.labels_[i]}')

    for cluster_number in sorted(list(set(clusterer.labels_))):
        print(f'Cluster: {cluster_number}' if cluster_number > -1 else 'Outliers')
        print([name for (name, cluster) in zip(names, clusterer.labels_) if cluster == cluster_number])

    mds_seed = np.random.RandomState(seed=0)
    mds = MDS(n_components=2, metric=True, max_iter=3000, eps=1e-12,
              dissimilarity="precomputed", n_jobs=1, random_state=mds_seed)
    pos = mds.fit_transform(pairwise_distances)

    # Rotate data so the the first point is in the 0, 1 position
    if pos[0, 0] == 0 and pos[0, 1] == 0:
        s, c = 0, 0
    elif pos[0, 0] == 0:
        s, c = 0, 1/pos[0, 1]
    else:
        s, c = pos[0, 0]/(pos[0, 0]**2+pos[0, 1]**2), pos[0, 1]/(pos[0, 0]**2+pos[0, 1]**2)
    if s == 0 and c == 0:
        pass
    else:
        rot = np.array([[c, -s], [s, c]])
        for i in range(np.shape(pos)[0]):
            pos[i] = rot @ pos[i]

    # Scale so all points fit in a unit circle
    pos /= np.max(np.sqrt(pos[:, 0]**2+pos[:, 1]**2))

    fig = matplotlib.pyplot.figure()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    fig = line_plot(pos[:, 0], pos[:, 1], text=names, font_settings={'size': 12},
                    x_lim=(-1.05, 1.05), y_lim=(-1.05, 1.05), y_ticks=[], x_ticks=[],
                    style='.', markersize=20, force_square=True,
                    colors_custom=[((0, 0, 0, 1) if label == -1 else
                                    matplotlib.colormaps['tab10'](label+1)) for label in clusterer.labels_],
                    show_plot_in_spyder=False, fig=canvas.figure)

    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()

    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    canvas.draw_idle()

    return


if __name__ == '__main__':
    from app import run_app
    run_app()
