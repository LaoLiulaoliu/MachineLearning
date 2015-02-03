#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>
#
# knn need to normalize the data, or else the feature data
# with large scale will dominate the distance calculation,
# then we only train this large scale feature.
#
# We can tuning the k, let classifier more accuracy.

from __future__ import print_function
from __future__ import division
from collections import Counter

import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__) + '..')
sys.path.append(parent_dir)
import toolkit

# It is a good idea not load such large library at once
import numpy as np


def load_data(fname, label=True, sep=None):
    data_mat, label_mat = [], []
    with open(fname) as fd:
        for line in fd:
            data = line.strip().split(sep)
            if label:
                data_mat.append( [float(i) for i in data[:-1]] )
                label_mat.append(data[-1])
            else:
                data_mat.append( [float(i) for i in data] )
    return np.mat(data_mat), np.mat(label_mat).T

def process_data(fname='datingTestSet.txt'):
    """
    """
    if not hasattr(process_data, '_loaded'):
        setattr(process_data, '_loaded', ())

        X, Y = load_data(fname, sep='\t')
        x_normalized, x_min, x_range = toolkit.normalizing(X)
        process_data._loaded = (x_normalized, x_min, x_range, Y)

    return process_data._loaded

def knn_classifier(one_data, x_normalized=None, x_min=None, x_range=None, Y=None, k=5):
    """ Euclidean distance of all features

    Every classification will calculate the distance will every data sample,
    it costs a lot of CPU time and Memory.

    :param one_data: type 1 x n matrix, data need to be classified
    :param k: type int, usually k is not greater than 20
    """
    if x_normalized is None and Y is None:
        x_normalized, x_min, x_range, Y = process_data()

    one_normalized = (one_data - x_min) / x_range
    distances = toolkit.euclidean_distance(x_normalized, one_normalized)

    distances = distances.T.A[0]
    distance_sort_indices = np.argsort(distances)

    class_counter = Counter()
    for i in range(k):
        class_counter[ Y[distance_sort_indices[i]][0, 0] ] += 1
    return class_counter.most_common(1)[0][0]


class test(object):
    """ the ratio can be tuned
    """
    def __init__(self, fname='datingTestSet.txt', sep='\t'):
        self.X, self.Y = load_data(fname=fname, sep=sep)

    def cross_validation(self, ratio=0.1):
        """ Usually use 10% random data as test data

        :param ratio: percentage of test data
        """
        m, n = np.shape(self.X)
        test_m = int(m * ratio)

        sample_indices = np.arange(m)
        np.random.shuffle(sample_indices)
        x_normalized, x_min, x_range = toolkit.normalizing(self.X[sample_indices[test_m:],])

        error, counter = 0, 0
        for i in sample_indices[:test_m]:
            y = knn_classifier(self.X[i], x_normalized, x_min, x_range, self.Y[sample_indices[test_m:],])
            if self.Y[i][0, 0] != y:
                error += 1
            counter += 1
        print('error rate is: {}'.format(error / counter))

if __name__ == '__main__':
    test().cross_validation()
    print( knn_classifier(np.mat([70843, 7.436056, 1.479856])) )
