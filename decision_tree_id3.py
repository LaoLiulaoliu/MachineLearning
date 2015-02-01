#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function
from __future__ import division

from collections import Counter
import numpy as np

def source_entropy(Y):
    m, _ = np.shape(Y)
    labels = Y.T.A[0]

    entropy = 0
    frequency = Counter()
    for value in set(labels):
        frequency[value] = np.sum(labels == value) / m

    for value, freq in frequency.iteritems():
        entropy -= freq * np.log2(freq)

    return entropy
