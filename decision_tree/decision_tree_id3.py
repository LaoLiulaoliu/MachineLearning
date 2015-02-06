#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function
from __future__ import division

from collections import Counter
import numpy as np

import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__) + '..')
sys.path.append(parent_dir)
import toolkit

def source_entropy(Y):
    """ The result of entropy is [0, ∞)
    """
    m, _ = np.shape(Y)
    labels = Y.flatten().A[0]

    entropy = 0
    counter = Counter()
    for value in labels: counter[value] += 1

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
        for value in data[:, i]: counter[value[0, 0]] += 1

        entropy = 0
        for val, times in counter.iteritems():
            branch_entropy = source_entropy( data[ (data[:, i].flatten().A[0] == val), -1 ] )
            entropy += times / n * branch_entropy

        if entropy < mini_entropy:
            mini_entropy = entropy
            best_feature = i
    return best_feature


def make_tree(data, labels):
    """ If the tree only have one class, this branch over.
        If the tree only have class column, this branch over.
    """
    _, n = np.shape(data)
    if len( set(data[:, -1].flatten().A[0]) ) == 1:
        return data[0, -1]
    if n == 1:
        counter = Counter()
        for value in data.flatten().A[0]: counter[value] += 1
        return counter.most_common(1)[0][0]

    best_feature = choose_best_feature(data)
    tree = { labels[best_feature]: {} }

    for value in set(data[:, best_feature].flatten().A[0]):
        lines = range(best_feature) + range(best_feature+1, n)
        tree[labels[best_feature]][value] = \
            make_tree(data[ data[:, best_feature].flatten().A[0] == value ][:, lines],
                      labels[lines])
    return tree


def dump_tree(trees):
    import cPickle
    with open('trees.txt', 'w') as fd:
        cPickle.dump(trees, fd)

def load_tree():
    import cPickle
    with open('trees.txt') as fd:
        return cPickle.load(fd)

if __name__ == '__main__':

    # 16 samples have missing feature values, denoted by "?"
    data, _ = toolkit.load_data('../breast-cancer-wisconsin.data', label=False, sep=',')
    labels = np.array(['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])
    r = make_tree(data[:, 1:], labels[1:])
    print(r)

    r = make_tree(np.mat([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]), np.array(['no surfacing','flippers', 'Class']))
    print(r)

