#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>
#
# knn need to normalize the data, or else the feature data
# with large scale will dominate the distance calculation,
# then we only train this large scale feature.

from __future__ import print_function
from __future__ import division
from collections import Counter
from toolkit import normalizing

# It is a good idea not load such large library at once
import numpy as np


def load_data(fname, label=True, sep=None):
    data_mat, label_mat = [], []
    with open(fname) as fd:
        for line in fd:
            data = line.strip().split(sep)
            if label:
                data_mat.append(data[:-1])
                label_mat.append(data[-1])
            else:
                data_mat.append(data)
    return np.mat(data_mat), np.mat(label_mat).T

def process_data():
    """
    """
    if not hasattr(process_data, '_loaded'):
        setattr(proess_data, '_loaded', ())

        X, Y = load_data(sep='\t')
        x_normalized, x_min, x_range = normalizing(X)
        process_data._loaded = (x_normalized, x_min, x_range, Y)

    return process_data._loaded

def knn_classifier(one_data, k=5):
    """ Euclidean distance of all features

    Every classification will calculate the distance will every data sample,
    it costs a lot of CPU time and Memory.

    :param one_data: type 1 x n matrix, data need to be classified
    :param k: type int, usually k is not greater than 20
    """
    x_normalized, x_min, x_range, Y = process_data()
    one_normalized = (one_data - x_min) / x_range
    distances = np.sqrt( np.power((x_normalized - one_normalized), 2).sum(axis=1) )

    distances = distances.T.A[0]
    distance_sort_indices = np.argsort(distances)

    class_counter = Counter()
    for i in range(k):
        class_counter[ Y[distance_sort_indices[i]][0, 0] ] += 1
    return class_counter.most_common(1)[0][0]

def cross_validation(ratio=0.1):
    """ Usually use 10% random data as test data

    :param ratio: percentage of test data
    """
    x_normalized, x_min, x_range, Y = process_data()
    m, n = np.shape(x_normalized)
    test_data_num = int(m * ratio)
    sample_indices = range(m)
    sample_indices = np.random.shuffle(sample_indices)

    for i in sample_indices[:test_data_num]:
        knn_classifier(x_normalized[i], 3)

