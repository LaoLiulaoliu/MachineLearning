#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import sys
import numpy as np

def read_input(fd):
    for line in fd:
        yield line.strip()

def mapper():
    input = [float(i) for i in read_input(sys.stdin)]
    input_num = len(input)
    input = np.mat(input)
    square_input = np.power(input, 2)
    print('{}\t{}\t{}\n'.format(input_num, np.mean(input), np.mean(square_input)))
    print('report: still alive', file=sys.stderr)

def reducer():
    mapper_out = [line.split('\t') for line in read_input(sys.stdin)]
    cumulative_num, cumulative_sum, cumulative_square = 0, 0, 0
    for item in mapper_out:
        num, mean_val, mean_square = map(float, item)
        cumulative_num += num
        cumulative_sum += num * mean_val
        cumulative_square += num * mean_square

    total_mean = cumulative_sum / cumulative_num
    total_variance = (cumulative_suqare + cumulative_num * total_mean**2 - 2 * total_mean * cumulative_sum) / cumulative_num
    print('{}\t{}\t{}\n'.format(cumulative_num, total_mean, total_variance))
    print('report: still alive', file=sys.stderr)

