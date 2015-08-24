#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

from toolkit import load_data, euclidean_distance
import numpy as np


def rand_k_centroids(data, k=3):
    m, n = np.shape(data)
    centroids = np.mat( np.zeros((k, n)) )
    for j in range(n):
        min_j   = np.min(data[:, j])
        range_j = np.max(data[:, j]) - min_j
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids

def kmeans(data, k, calculate_distance=euclidean_distance, create_centroids=rand_k_centroids):
    """ k means only converge to local minimum,
        the result will easily affect by initial centroids

        two problems:
            1. one cluster can split. which one?
                a. the cluster with biggest SSE(sum of squared error) until cluster growth to k;
                b. after the cluster splitting, sum of all clusters' SSE is minimum
            2. two cluster can merge. which two?
                a. two nearest centroids;
                b. after the two centroids merging, sum of all clusters' SSE is minimum

        After analysing the two problems, we got bisecting k means.
    """
    m, n = np.shape(data)
    # initial with -1, in case the compare changed conflict
    # 簇分配矩阵 [分配簇编号，和分配簇Sum of Squared Error]
    cluster_assignment = mp.mat( -np.ones(m, 2) )
    changed = True

    centroids = create_centroids(data, k)
    while changed:
        changed = False

        for i in range(m):
            min_distance = -1; min_index = -2

            for j in range(k):
                distance = calculate_distance( data[i, :], centroids[j, :] )
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance; min_index = j

            if cluster_assignment[i, 0] != min_index: changed = True
            cluster_assignment[i, :] = (min_index, min_distance**2)

        for point in range(k):
            points_in_cluster = data[ np.nonzero(cluster_assignment[:, 0].A == point)[0] ]
            centroids[point, :] = np.mean(points_in_cluster, axis=0)
    return centroids, cluster_assignment


def bisecting_kmeans(data, k, calculate_distance=euclidean_distance):
    """ We start with one cluster, split it to k.
        Split the cluster which can decrease the SSE most.
    """
    m, n = np.shape(data)


if __name__ == '__main__':
    data, _ = load_data('data.txt', label=False, sep='\t', float)
    kmeans(data, 3)

