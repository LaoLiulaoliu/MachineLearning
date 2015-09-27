#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import numpy as np
import numpy.linalg as LA


def svd(X, ratio=0.9):
    U, Sigma, Vt = LA.svd(X) # m*m, ndarray, n*n
    idx = 0; proportion = 0
    for i, sigma in enumerate(np.power(Sigma, 2) / sum(np.power(Sigma, 2))):
        proportion += sigma
        if proportion > ratio:
            idx = i
            break

    diagonal_sigma = np.mat(np.eye(idx) * Sigma[:idx])
    X_reduction = X.T * U[:, :idx] * diagonal_sigma.I # n*idx
    # reconstruct_X = U[:, :idx] * diagonal_sigma * Vt[:idx, :]
    return X_reduction

