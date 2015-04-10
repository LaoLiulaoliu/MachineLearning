#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>
#
# Stepwise regression analysis the important features,
# stop collecting the inconsequential features timely.

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from linear_regression import load_data, feature_normalize, normal_equation, batch_gradient_descent

def stepwise(X, Y, epsilon=0.01, iteration=200):
    m, n = np.shape(X)
    X_norm, X_mean, X_std = feature_normalize(X)
    thetas = np.ones((iteration, n))
    theta = np.zeros((n, 1))
    fit_theta = np.zeros((n, 1))
    for i in range(iteration):

        minimum_error = -1
        for j in range(n):
            for sign in [-1, 1]:
                theta_test = theta.copy()
                theta_test[j] += epsilon * sign
                error = np.power(X_norm * theta_test - Y, 2).sum()
                if minimum_error < 0 or error < minimum_error:
                    minimum_error = error
                    fit_theta = theta_test
        theta = fit_theta
        thetas[i] = theta.T
    return thetas


def theta_iteration(thetas):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thetas)
    plt.show()

if __name__ == '__main__':
    data, label = load_data('linear_data_3d.txt')
    thetas = stepwise(data, label, 0.009, 6000)
    for i in thetas:
        print(i)

    X_norm, X_mean, X_std = feature_normalize(data)
    print('normal equation theta {}'.format(normal_equation(X_norm, label).T))

    theta = np.zeros((data.shape[1], 1))
    theta, cost_history = batch_gradient_descent(X_norm, label, theta, 0.001, 6000)
    print('gradient descent theta {}'.format(theta.T))

    theta_iteration(thetas)
