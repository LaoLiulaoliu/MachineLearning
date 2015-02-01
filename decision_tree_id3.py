#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function
from __future__ import division

from collections import Counter
import numpy as np

def source_entropy(Y):
    """ The result of entropy is [0, ∞)
    """
    m, _ = np.shape(Y)
    labels = Y.flatten().A[0]

    entropy = 0
    frequency = Counter()
    for value in labels:
        frequency[value] += 1

    for value, freq in frequency.iteritems():
        entropy -= freq * np.log2(freq)

    return entropy
