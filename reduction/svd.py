#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import numpy as np
import numpy.linalg as LA

def euclid_distance(A, B):
    """ normalization of Euclid distance to (0, 1]

        return: 0 means A, B are very far, 1 means A == B
    """
    return 1. / (1 + LA.norm(A - B))

def pearson_correlation(A, B):
    """ not sensitive to magnitude, e.g. pearson([1,1]) == pearson([5,5])

        return: 0 means A, B are very far, 1 means A == B
    """
    if len(A) < 3: return 1
    return 0.5 + 0.5 * np.corrcoef(A, B, rowvar=0)

def cosine_similarity(A, B):
    """ If positive and negative numbers are in A or B, cosine(A, b) range in [-1, 1],
        0.5 + 0.5 * cosine(A, B) normalize result to [0, 1]
        Only positive number in A and B, cosine(A, B) range in [0.5, 1]

        return: 0 means A, B are very far, 1 means A == B
    """
    cosine = sum(A * B) / (LA.norm(A) * LA.norm(B))
    return 0.5 + 0.5 * cosine


def svd():
    pass


def recommend(X):
    m, n = np.shape(X)
