# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:04:35 2024

@author: nicol
"""

import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import tempfile
import argparse
import sys
import argparse
import numpy as np
import numpy.ma as ma

topology = '/hdnas/marcol/tau/polimeri-120/tetramer/select/QT-clustering/my100/select_CA.pdb';
trajectory = '/hdnas/marcol/tau/polimeri-120/tetramer/select/QT-clustering/my100/traj_100ps.xtc';
t = md.load(trajectory,top=topology);
n_frames = t.n_frames;
#sel = t.topology.select('protein');
sel = t.topology.select('all');
t = t.atom_slice(sel);

# tempfile = tempfile.NamedTemporaryFile();
# # distances = np.memmap(tempfile.name, dtype=float, shape=(n_frames,n_frames));
# distances = np.empty((n_frames, n_frames), dtype=float);
# t.center_coordinates();
# for i in range(n_frames):
#     distances[i] = md.rmsd(target=t, reference=t, frame=i, precentered=True);
    
# # t = None;
# cutoff_mask = distances <= 0.4;
# # distances = None;
# centers = [];
# cluster = 0;
# labels = np.empty(n_frames);
# labels.fill(np.NAN);

# while cutoff_mask.any():
#     membership = cutoff_mask.sum(axis=1)
#     center = np.argmax(membership)
#     members = np.where(cutoff_mask[center,:]==True)
#     if max(membership) <= 2:
#         labels[np.where(np.isnan(labels))] = -1
#         break
#     labels[members] = cluster
#     centers.append(center)
#     cutoff_mask[members,:] = False
#     cutoff_mask[:,members] = False
#     cluster = cluster + 1


N = t.n_frames
matrix = np.ndarray((N, N), dtype=np.float16);
for i in range(N):
    rmsd_ = md.rmsd(target=t, reference=t, frame=i, precentered=True)*10.0;
    matrix[i] = rmsd_;
print('>>> Calculation of the RMSD matrix completed <<<')
cutoff = 4.0;
# ---- Delete unuseful values from matrix (diagonal &  x>threshold) -----------
matrix[matrix > cutoff] = np.inf;
matrix[matrix == 0] = np.inf;
# matrix[matrix < 0.001] = np.inf;
degrees = (matrix < np.inf).sum(axis=0);


clusters_arr = np.ndarray(N, dtype=np.int64)
clusters_arr.fill(-1)

min_elem = 5;
ncluster = 0
while True:
    # This while executes for every cluster in trajectory ---------------------
    len_precluster = 0
    while True:
        # This while executes for every potential cluster analyzed ------------
        biggest_node = degrees.argmax()
        precluster = []
        precluster.append(biggest_node)
        candidates = np.where(matrix[biggest_node] < np.inf)[0]
        next_ = biggest_node
        distances = matrix[next_][candidates]
        while True:
            # This while executes for every node of a potential cluster -------
            next_ = candidates[distances.argmin()]
            precluster.append(next_)
            post_distances = matrix[next_][candidates]
            mask = post_distances > distances
            distances[mask] = post_distances[mask]
            if (distances == np.inf).all():
                break
        degrees[biggest_node] = 0
        # This section saves the maximum cluster found so far -----------------
        if len(precluster) > len_precluster:
            len_precluster = len(precluster)
            max_precluster = precluster
            max_node = biggest_node
            degrees = ma.masked_less(degrees, len_precluster)
        if not degrees.max():
            break
    # General break if minsize is reached -------------------------------------
    if len(max_precluster) < min_elem:
        break

    # ---- Store cluster frames -----------------------------------------------
    clusters_arr[max_precluster] = ncluster
    ncluster += 1
    print('>>> Cluster # {} found with {} frames at center {} <<<'.format(
          ncluster, len_precluster, max_node))

    # ---- Update matrix & degrees (discard found clusters) -------------------
    matrix[max_precluster, :] = np.inf
    matrix[:, max_precluster] = np.inf

    degrees = (matrix < np.inf).sum(axis=0)
    if (degrees == 0).all():
        break

np.savetxt('/hdnas/marcol/tau/polimeri-120/tetramer/select/QT-clustering/my100/QT_Clusters.txt', clusters_arr, fmt='%i')

# NMRcluster format. VMD interface
with open('/hdnas/marcol/tau/polimeri-120/tetramer/select/QT-clustering/my100/QT_Visualization.log', 'wt') as clq:
    for numcluster in np.unique(clusters_arr):
        clq.write('{}:\n'.format(numcluster))
        members = ' '.join([str(x + 1)
                            for x in np.where(clusters_arr == numcluster)[0]])
        clq.write('Members: ' + members + '\n\n')


clusters_arr.sort(); clusters = list(np.array(clusters_arr));
unique = list(set(clusters_arr));
count = [];
for i in unique:
    count.append(clusters.count(i));

import pandas as pd
df = pd.DataFrame(); df["index"] = unique; df["counts"] = count;

np.savetxt(r'/hdnas/marcol/tau/polimeri-120/tetramer/select/QT-clustering/my100/clusters_counts.txt', df.values, fmt='%d');












    
