# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:46:51 2024

@author: KOlson
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering  # , HDBSCAN
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram
import hdbscan


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


with open(r'C:\Users\KOlson\OneDrive - Metrolaser, Inc\Documents\Projects\MD05\Data\LARS\Brackets\pair_results.pkl', 'rb') as f:
    pair_results = pickle.load(f)
points = []
for pr in pair_results:
    points.append(pr['names'][0])
    points.append(pr['names'][1])
points = sorted(list(set(points)))
dims = len(points)-1

maxq = 0
for pr in pair_results:
    maxq = max(maxq, pr['quality'])

d = np.zeros((dims+1, dims+1))
for pr in pair_results:
    i0 = points.index(pr['names'][0])
    i1 = points.index(pr['names'][1])
    d[i0, i1] = d[i1, i0] = 1-pr['match_probability']  # + pr['quality']/maxq

n_clusters = 8

model = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='single', compute_distances=True)
labels = model.fit_predict(d)

print('Agglomerative Clustering')
points = np.array(points, dtype=str)
clustered_points = [[]]*n_clusters
for i in range(n_clusters):
    clustered_points[i] = points[labels == i]

for p in sorted(clustered_points, key=lambda x: x[0]):
    print(p)

# model2 = HDBSCAN(min_cluster_size=2, min_samples=2, metric='precomputed')
# result2 = model2.fit(d)
# labels2 = model2.fit(d).labels_
# probs2 = model2.fit(d).probabilities_

# print('HDBSCAN')
# unique_labels = sorted(list(set(labels2)))

# clustered_points = [[]]*len(unique_labels)
# clustered_probs = [[]]*len(unique_labels)
# for i, lab in enumerate(unique_labels):
#     clustered_points[i] = points[labels2 == lab]
#     clustered_probs[i] = probs2[labels2 == lab]

# for pt, pr, lab in sorted(zip(clustered_points, clustered_probs, unique_labels), key=lambda x: x[0][0]):
#     print(lab, pt)
#     print(pr)

# fig = plt.figure()
# plt.title("Agglomerative Clustering Dendrogram")
# fig.set_size_inches(10, 5)
# # plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode="level", labels=points, above_threshold_color='k')
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()


# Create HDBSCAN clusterer
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='precomputed')
clusterer.fit(d)
print(clusterer.labels_)

# Plot the dendrogram
# fig = plt.figure()
# fig.set_size_inches(10, 50)
# ax = plt.gca()
# clusterer.condensed_tree_.plot(axis=ax, select_clusters=True,
#                                selection_palette=['r', 'b', 'g', 'y', 'm', 'c'],
#                                label_clusters=True, log_size=False, cmap='gist_ncar')
# tree = clusterer.condensed_tree_.to_numpy()
# plt.show()

nmds = MDS(n_components=2, metric=True, max_iter=3000, eps=1e-12, dissimilarity="precomputed", n_jobs=1)
pos = nmds.fit_transform(d**2)
com = np.mean(pos[:, 0]), np.mean(pos[:, 1])
pos[:, 0] -= com[0]
pos[:, 1] -= com[1]
pos /= np.max(np.sqrt(pos[:, 0]**2+pos[:, 1]**2))
fig = plt.figure(figsize=(6, 6))
ax = fig.gca()
ax.scatter(pos[:, 0], pos[:, 1])
ax.set_xlim((-1.05, 1.05))
ax.set_ylim((-1.05, 1.05))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
for i, t in enumerate(points):
    ax.annotate(" "+t, (pos[i, 0], pos[i, 1]))
plt.show()
