#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__) + './../')
sys.path.append(parent_dir)

from toolkit import load_data, euclidean_distance
import numpy as np


def rand_k_centroids(data, k=3):
    m, n = np.shape(data)
    centroids = np.mat( np.zeros((k, n)) )

    min_j = np.min(data, axis=0)
    rang_j = np.max(data, axis=0) - min_j

    centroids = np.tile(min_j, (k, 1)) + \
        np.multiply(np.tile(rang_j, (k, 1)),
                    np.random.rand(k, n))

#    for j in range(n):
#        min_j   = np.min(data[:, j])
#        range_j = np.max(data[:, j]) - min_j
#        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids

def kmeans(data, k, calculate_distance=euclidean_distance, create_centroids=rand_k_centroids):
    """ k means only converge to local minimum,
        the result will easily affect by initial centroids
        Theoretically, the result of clustering will shake, but happens rarely.

        two problems:
            1. one cluster can split. which one?
                a. the cluster with biggest SSE(sum of squared error) until cluster growth to k;
                b. after the cluster splitting, sum of all clusters' SSE is minimum
            2. two cluster can merge. which two?
                a. two nearest centroids;
                b. after the two centroids merging, sum of all clusters' SSE is minimum

        After analysing the two problems, we got bisecting k means.

    :return centroids: k centroids coordinate
    :return cluster_assignment: [i, j], i is the classification, j is the distance from its centroid.
    :rtype: typle
    """
    m, n = np.shape(data)
    # initial with -1, in case the compare changed conflict
    # 簇分配矩阵 [assign data points to a centroid, holds Sum of Squared Error to each point]
    cluster_assignment = np.mat( -np.ones((m, 2)) )
    centroids = create_centroids(data, k)
    changed = True

    while changed:
        changed = False

        for i in range(m): # for each data point, assign it to the closest centroid
            min_distance = -1; min_index = -2

            for j in range(k):
                distance = calculate_distance( data[i, :], centroids[j, :] )
                if distance < min_distance or min_distance == -1:
                    min_distance = distance; min_index = j

            if cluster_assignment[i, 0] != min_index: changed = True
            cluster_assignment[i, :] = (min_index, min_distance**2)

        for centre in range(k): # recalculate centroids
            points_in_cluster = data[ np.nonzero(cluster_assignment[:, 0].A == centre)[0] ] # get all points in this cluster
            # assign centroid to mean of all points in this cluster
            # if points_in_cluster is empty, this centroid is np.nan, warnings is printed
            centroids[centre, :] = np.mean(points_in_cluster, axis=0)
    return centroids, cluster_assignment


def bisecting_kmeans(data, k, calculate_distance=euclidean_distance):
    """ We start with one cluster, split it to k.
        Split the cluster which can decrease the SSE(sum of squared error) most.
        converge at global minimum
    """
    m, n = np.shape(data)
    centroid = np.mean(data, axis=0).tolist()[0]
    centroids = [centroid] # list with one centroid

    cluster_assignment = np.mat( np.zeros((m, 2)) )
    for i in range(m): # calculate initial SSE
        cluster_assignment[i, 1] = calculate_distance(centroid, data[i, :]) ** 2

    while(len(centroids) < k):
        lowest_SSE = -1
        for i, _ in enumerate(centroids):
            subcluster = data[np.nonzero(cluster_assignment[:, 0].A == i)[0], :] # get all points in cluster i
            # this cluster has no point, if deleted this centroid, centroids length may never reach k
            # if len(subcluster) == 0: del(centroids[i]); continue

            subcentroids, subcluster_assignment = kmeans(subcluster, 2, calculate_distance)
            if np.any( np.isnan(subcentroids) ) == True:
                # Do 2means again if one cluster has no point, or else will generate zero point centroid
                # that means this subcluster don't need to be split
                subcentroids, subcluster_assignment = kmeans(subcluster, 2, calculate_distance)
                if np.isnan( np.sum(subcentroids) ) == True:
                    continue

            subSSE = np.sum( subcluster_assignment[:, 1] )
            non_subSSE = np.sum( cluster_assignment[np.nonzero(cluster_assignment[:, 0].A != i)[0], 1] )
            if (subSSE + non_subSSE) < lowest_SSE or lowest_SSE == -1:
                lowest_SSE = subSSE + non_subSSE
                best_split_centre = i
                best_subcentroids = subcentroids
                best_subcluster_assignment = subcluster_assignment

        if lowest_SSE == -1: break # no suitable split in centroids
        print("{} centroids SSE: {}".format(len(centroids)+1, lowest_SSE))
        # len(centroids) larger than any index in centroids, need use to assign first
        # if best_split_centre is 0, or len(centroids) is 1, they can affect best_subcluster_assignment without intermediate value
        row1 = np.nonzero(best_subcluster_assignment[:, 0].A == 1)[0]
        row0 = np.nonzero(best_subcluster_assignment[:, 0].A == 0)[0]
        best_subcluster_assignment[row1, 0] = len(centroids)
        best_subcluster_assignment[row0, 0] = best_split_centre
        cluster_assignment[np.nonzero(cluster_assignment[:, 0].A == best_split_centre)[0], :] = best_subcluster_assignment

        centroids[best_split_centre] = best_subcentroids[0].tolist()[0] # replace a centroid with two better centroids
        centroids.append(best_subcentroids[1].tolist()[0])
    return np.mat(centroids), cluster_assignment


if __name__ == '__main__':
    data, _ = load_data('kmeans.data', label=False, sep='\t', func=float)

    cents, cluster_assignment = kmeans(data, 3)
    centroids = [[-0.02298687,  2.99472915],
                 [-3.38237045, -2.9473363 ],
                 [ 2.8692781,  -2.54779119]]
    print( np.sum(cents - centroids) )

    print( bisecting_kmeans(data, 10)[0] )

