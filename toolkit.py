#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import numpy as np

def standardizing(X):
    x_mean = np.mean(X, axis=0)
    x_std  = np.std (X, axis=0)

    x_standard = (X - x_mean) / x_std
    return x_standardized, x_mean, x_std

def normalizing(X):
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_range = x_max - x_min

    x_normalized = (X - x_min) / x_range
    return x_normalized, x_min, x_range

def euclidean_distance(A, B, axis=None):
    """ A, B are array or matrix
    """
    return np.sqrt( np.power(A - B, 2).sum(axis=axis) )


def load_data(fname, label=True, sep=None, func=lambda x: x):
    data_mat, label_mat = [], []
    with open(fname) as fd:
        for line in fd:
            if '?' in line: continue
            data = line.strip().split(sep)
            if label:
                data_mat.append( [func(i) for i in data[:-1]] )
                label_mat.append(data[-1])
            else:
                data_mat.append( list(map(func, data)) )
    return np.mat(data_mat), np.mat(label_mat).T

