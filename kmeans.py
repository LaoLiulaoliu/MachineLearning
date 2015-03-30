#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function
from __future__ import division

from toolkit import load_data, euclidean_distance
import numpy as np


def rand_k_centroids(data, k=3):
    m, n = np.shape(data)
    centroids = np.mat( np.zeros((k, n)) )
    for j in range(n):
        min_j = np.min(data[:, j])
        max_j = np.max(data[:, j])
        centroids[:, j] = min_j + (max_j - min_j) * np.random.rand(k, 1)
    return centroids

def kmeans(data, k, calculate_distance=euclidean_distance, create_centroids=rand_k_centroids):
    m, n = np.shape(data)
    indices = mp.mat( -np.ones(m, 2) ) # not initial with 0, in case the compare changed conflict
    changed = True

    centroids = create_centroids(data, k)
    while changed:
        changed = False

        for i in range(m):
            min_distance = -1; min_index = -1

            for j in range(k):
                distance = calculate_distance( data[i, :], centroids[j, :] )
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance; min_index = j

            indices[i, :] = (min_index, min_distance**2)
            if indices[i, 0] != min_index:
                changed = True

        for centre in range(k):
            points = data[ np.nonzero(indices[:, 0].A == centre)[0] ]
            centroids[centre, :] = np.mean(points, axis=0)
    return centroids

if __name__ == '__main__':
    data, _ = load_data('data.txt', label=False, sep='\t', float)
    kmeans(data, 3)

