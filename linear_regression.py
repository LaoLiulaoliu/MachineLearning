#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function

import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def load_data(fname='data.txt'):
    data_array, label_array = [], []
    with open(fname) as fd:
        for line in fd:
            line = line.strip().split()
            data_array.append( [1.] + [float(i) for i in line[:-1]] )
            label_array.append( float(line[-1]) )
    return np.mat(data_array), np.mat(label_array).T


def normal_equation(X, Y):
    """ normal equation do not need feature scaling
    fit for feature less than 100000.

    shape(X) -> (m, n)
    shape(Y) -> (m, 1)
    [(n, m) * (m, n)]' * (n, m) * (m, 1) -> (n, 1)
    """
    theta = (X.T * X)
    if np.linalg.det(theta) == 0:
        print("numpy.linalg.linalg.LinAlgError: Singular matrix' not reversible")
        return []
    theta = theta.I * X.T * Y
    return theta

def locally_weighted_linear_regression(data_item, X, Y, k=0.8):
    """
    weighted: square matrix, give every data item a different weight.
    """
    m, n = np.shape(X)
    weighted = np.eye(m)
    np.exp( -.5 * (data_item - X) / k**2 )

def feature_normalize(X):
    """ `var` replace `std` also works
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[0, 0] = 1
    X_normalize = (X - X_mean) / X_std
    X_normalize[:, 0] = 1.0
    return X_normalize, X_mean, X_std

def feature_scaling(X):
    """ linear regression with feature scaling, will use mean, std in test data.

    * Feature scaling in gradient descent will train different parameters,
    compare to normal equation with no feature scaling.
    * If we have more than one feature, we need feature scaling.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[0, 0] = 1

    m, n = np.shape(X)
    norm_data = X - np.tile(X_mean, (m, 1))
    norm_data = norm_data / np.tile(X_std, (m, 1))
    norm_data[:, 0] = 1.0
    return norm_data, X_mean, X_std

def cost_function(X, Y, theta):
    m, n = np.shape(X)
    cost = X * theta - Y # cost.shape = (m, 1)
    J = 0.5 / m * cost.T * cost
    return J

def plot_cost_function(X, Y):
    """ This picture shows that we cat get global minimum of J(θ)
        by partial derivative J(θ) on θ
    """
    upper_bound = 2* X.max()
    lower_bound = 2* X.min()
    theta0_vals = np.linspace(lower_bound, upper_bound, 100)
    theta1_vals = np.linspace(lower_bound, upper_bound, 100)
    len_0, len_1 = len(theta0_vals), len(theta1_vals)
    J_vals = np.zeros((len_0, len_1))

    for i in xrange(len_0):
        for j in xrange(len_1):
            theta = np.mat( [1, theta0_vals[i], theta1_vals[j]] ).T
            J_vals[i, j] = cost_function(X, Y, theta)

    fig = plt.figure()
    fig.suptitle("cost function J change on theta")
    ax = fig.add_subplot(111, projection='3d')
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    surf = ax.plot_surface( theta0_vals, theta1_vals, J_vals,
                            rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False )
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('cost value')
    plt.show()


def batch_gradient_descent(X, Y, theta, alpha=0.01, iters=1000):
    """ gradient descent to get local minimum of cost function

    :param theta: theta is an array
    :param iters: iters times
    """
    m, n = np.shape(X)
    cost_history = np.zeros((iters, 1))

    for i in xrange(iters):
        theta -= alpha/m * (X.T * (X * theta - Y))
        cost_history[i] = cost_function(X, Y, theta)
        if cost_history[i] < 0.001: break
    return theta, cost_history

def tune_learning_rate(cost_history):
    length = len(cost_history)
    plt.figure()
    plt.subplot(111)
    plt.plot(range(length), cost_history, 'r-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost value')
    plt.title('Different alpha have different learning rate')
    plt.show()

def plot_result(X, Y, theta, X_norm=None):
    """ X only have two column variables
    """
    if X_norm is not None:
        X = X_norm
    a, b, c = X[:, 1].getA(), X[:, 2].getA(), Y.getA()

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(a)

    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(a, b, c, c='r', marker='o')

    p, q = np.meshgrid(a, b)
    surf = ax.plot_surface( p, q, (X * theta).getA() )
    ax.set_xlabel('X1 label')
    ax.set_ylabel('X2 label')
    ax.set_zlabel('Y label')

    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Use to draw")
    parser.add_argument('--cost', '-c', help="plot cost function", action="store_true")
    parser.add_argument('--normal', '-n', help="calculate normal equation", action="store_true")
    parser.add_argument('--learn', '-l', help="plot one learning rate's convergence")
    option = parser.parse_args()

    data, label = load_data('data.txt')
    data_norm, data_mean, data_std = feature_scaling(data)
    # initialize theta as zero
    theta = np.zeros((np.shape(data_norm)[1], 1))

    print(option.cost, option.normal, option.learn)
    if option.cost:
        plot_cost_function(data, label)
    if option.normal:
        theta_n = normal_equation(data, label)
        print('normal equation solve theta: {}\n'.format(theta_n))
    if option.learn:
        theta, cost_history = batch_gradient_descent(data_norm, label, theta)
        tune_learning_rate(cost_history)
        plot_result(data, label, theta, data_norm)


if __name__ == '__main__':
    main()

