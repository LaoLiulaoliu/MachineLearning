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
    counter = Counter()
    for value in labels:
        counter[value] += 1

    for value, count in counter.iteritems():
        frequency = count / m
        entropy -= frequency * np.log2(frequency)

    return entropy


