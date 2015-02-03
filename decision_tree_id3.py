#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function
from __future__ import division

from collections import Counter
import numpy as np

import toolkit

def source_entropy(Y):
    """ The result of entropy is [0, ∞)
    """
    m, _ = np.shape(Y)
    labels = Y.flatten().A[0]

    entropy = 0
    counter = Counter()
    for value in labels:
        counter[value] += 1

    for value, count in counter.iteritems():
        frequency = count / m
        entropy -= frequency * np.log2(frequency)

    return entropy

def choose_best_feature(data):
    """ Choose the feature which hava largest information gain,
    entropy of source tree minus this branch's source entropy.
    Which means we need to choose the smallest source entropy branch.

    The last column of data is Class.
    """
    m, n = np.shape(data)
    mini_entropy = 10000.
    best_feature = -1

    for i in range(n-1):
        counter = Counter()
        for value in data[:, i]: counter[value] += 1

        entropy = 0
        for val, times in counter.iteritems():
            branch_entropy = source_entropy( data[ (data[:, i].T.A[0] == val), -1 ] )
            entropy += times / n * branch_entropy

        if entropy < mini_entropy:
            mini_entropy = entropy
            best_feature = i
    return best_feature


def make_tree(data, labels):
    choose_best_feature(data)



if __name__ == '__main__':

    # 16 samples have missing feature values, denoted by "?"
    data, _ = toolkit.load_data('breast-cancer-wisconsin.data', label=False, sep=',')
    labels = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    make_tree(data[:, 1:], labels[1:])

