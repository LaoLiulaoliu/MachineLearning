#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import numpy as np
import numpy.linalg as LA

def euclid_distance(A, B):
    """ normalization of Euclid distance
    """
    return 1. / (1 + LA.norm(A - B))

def pearson_distance(A, B):
    np.corrcoef(A, B, rowvar=0)

def svd():
    pass

def recommend(X):
    pass
