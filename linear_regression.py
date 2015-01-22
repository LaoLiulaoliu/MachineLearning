#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from __future__ import print_function
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def load_data(fname='data.txt'):
    with open(fname) as fd:
        N = len( fd.readline().strip().split() )

    data_array = []
    label_array = []
    with open(fname) as fd:
        for line in fd:
            line = line.strip().split()
            data_array.append( [1.0] + [float(i) for i in line[:N-1]] )
            label_array.append( float(line[-1]) )
    return mat(data_array), mat(label_array).T


def normal_equation(X, Y):
    """ normal equation do not need feature scaling
    fit for feature less than 100000.

    shape(X) -> (m, n)
    shape(Y) -> (m, 1)
    [(n, m) * (m, n)]' * (n, m) * (m, 1) -> (n, 1)
    """
    theta = (X.T * X)
    if linalg.det(theta) == 0:
        print("numpy.linalg.linalg.LinAlgError: Singular matrix' not reversible")
        return []
    theta = theta.I * X.T * Y
    return theta


def feature_normalize(X):
    """ `var` replace `std` also works
    """
    X_mean = mean(X, axis=0)
    X_std = std(X, axis=0)
    X_std[0, 0] = 1
    X_normalize = (X - X_mean) / X_std
    X_normalize[:, 0] = 1.0
    return X_regular, X_mean, X_std

def feature_scaling(X):
    """ linear regression with feature scaling, will use mean, std in test data.

    * Feature scaling in gradient descent will train different parameters,
    compare to normal equation with no feature scaling.
    * If we have more than one feature, we need feature scaling.
    """
    X_mean = mean(X, axis=0)
    X_std = std(X, axis=0)
    X_std[0, 0] = 1

    m, n = shape(X)
    norm_data = X - tile(X_mean, (m, 1))
    norm_data = norm_data / tile(X_std, (m, 1))
    norm_data[:, 0] = 1.0
    return norm_data, X_mean, X_std

def cost_function(X, Y, theta):
    m, n = shape(X)
    cost = X * theta - Y # cost.shape = (m, 1)
    J = 0.5 / m * cost.T * cost
    return J

def plot_cost_function(X, Y):
    """ This picture shows that we cat get global minimum of J(θ)
        by partial derivative J(θ) on θ
    """
    theta0_vals = linspace(-100, 100, 100)
    theta1_vals = linspace(-100, 100, 100)
    len_0, len_1 = len(theta0_vals), len(theta1_vals)
    J_vals = zeros((len_0, len_1))

    for i in xrange(len_0):
        for j in xrange(len_1):
            theta = mat( [1, theta0_vals[i], theta1_vals[j]] ).T
            J_vals[i, j] = cost_function(X, Y, theta)

    fig = plt.figure()
    fig.suptitle("cost function J change on theta")
    ax = fig.add_subplot(111, projection='3d')
    theta0_vals, theta1_vals = meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('cost value')
    plt.show()


def batch_gradient_descent(X, Y, theta, alpha, iters):
    """ gradient descent to get local minimum of cost function

    :param theta: theta is an array
    :param iters: iters times
    """
    m, n = shape(X)
    cost_history = zeros((iters, 1))

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

def plot_result(X, Y, theta):
    a, b, c = X[:, 1].getA(), X[:, 2].getA(), Y.getA()

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(a)


    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(a, b, c, c='r', marker='o')

    p, q = meshgrid(a, b)
    surf = ax.plot_surface( p, q, (X * theta).getA(),
                            rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False )
    ax.set_xlabel('X1 label')
    ax.set_ylabel('X2 label')
    ax.set_zlabel('Y label')

    ax.set_zlim(-500, 500)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()

if __name__ == '__main__':
    data, label = load_data()
    theta_n = normal_equation(data, label)
    print('normal equation solve theta: {}\n'.format(theta_n))

    data_norm, data_mean, data_std = feature_scaling(data)
    if None: plot_cost_function(data_norm, label)

    theta = zeros((shape(data_norm)[1], 1))
    alpha = 0.01
    iterations = 1000
    theta, cost_history = batch_gradient_descent(data_norm, label, theta, alpha, iterations)

    if 0: tune_learning_rate(cost_history)

    if 0: plot_result(data, label, theta)
