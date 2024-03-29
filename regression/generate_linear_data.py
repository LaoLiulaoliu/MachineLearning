#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import itertools

NUM = 10

def function(x, y):
    """三维函数，但可以忽略y 只用两个维度 """
    return 3 + 1.2 * x + 5 * y + 0.4 * np.cos(10 * x) + 0.6 * np.random.normal(0, 1, np.shape(x))

def plot():

    lin_x = np.arange(-NUM, NUM, 0.25)
    lin_y = function(lin_x, np.zeros(lin_x.shape))

    X, Y = [], []
    for i in itertools.combinations(np.arange(-NUM, NUM, 1), 2):
        X.append(i[0])
        Y.append(i[1])
    X, Y = np.array(X, copy=False), np.array(Y, copy=False)
    Z = function(X, Y)

    fig = plt.figure()
    fig.suptitle(r"TeX $z = 3 + 1.2 x + 5 y + 0.4 cos(10 x) + 0.6 N(0, 1)$")

    # 数据点的分布
    ax = fig.add_subplot(2, 2, 1)
    ax.hist(lin_x, histtype='bar')

    # 误差的分布, 因为是公式制造的误差，所以部分系统误差满足高斯分布，另一部分是余弦函数
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(lin_x, 0.4*np.cos(10*lin_x)+0.6*np.random.normal(0,1,lin_x.shape))

    # 二维情况下，线性函数对带有噪声数据点的拟合示意
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(lin_x, lin_y, s=2, c='red', marker='o')
    ax.plot(lin_x, 3+1.2*lin_x, 'g')
    ax.grid(True)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(X, Y, Z, c='r', marker='o')
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    ax.set_xlim(-NUM, NUM)
    ax.set_ylim(-NUM, NUM)

    plt.show()

def generate_data():
    with open('linear_data_3d.txt', 'w') as fd:
        for i in range(200):
            x = np.random.random_sample() * 2*NUM - NUM
            y = np.random.random_sample() * 2*NUM - NUM
            z = function(x, y)
            fd.write('{}\t{}\t{}\n'.format(x, y, z))

    with open('linear_data_2d.txt', 'w') as fd:
        for i in range(200):
            x = np.random.random_sample() * 2*NUM - NUM
            y = 0
            z = function(x, y)
            fd.write('{}\t{}\n'.format(x, z))


if __name__ == '__main__':
    if 0: generate_data()
    plot()
