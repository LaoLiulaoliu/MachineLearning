#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def load_data(fname='linear_data_3d.txt'):
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

def feature_normalize(X):
    """
    """
    X_min = np.min(X, axis=0)
    X_range = np.max(X, axis=0) - X_min
    X_normalize = (X - X_min) / X_range
    X_normalize[:, 0] = 1.0
    return X_normalize, X_min, X_range

def feature_standardize(X):
    """ `var` replace `std` also works
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[0, 0] = 1
    X_standardize = (X - X_mean) / X_std
    X_standardize[:, 0] = 1.0
    return X_standardize, X_mean, X_std

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


############ locally weighted linear regression
# 随着核k 的减小，训练误差逐渐变小，出现过拟合现象。测试误差在增大。
# 要找到训练误差减小，测试误差还没开始呈指数级增长的k，抓住数据潜在模式
def locally_weighted_linear_regression(data_item, X, Y, k=0.8):
    """
    weighted: square matrix, give every data item a different weight.
    theta: n x 1
    """
    m, n = np.shape(X)
    weighted = np.eye(m)
    for i in range(m):
        difference = data_item - X[i] 
        weighted[i, i] = np.exp( -.5 * difference * difference.T / k**2 )

#    print(weighted) # most element in weighted is 0
    theta = X.T * weighted * X
    if np.linalg.det(theta) == 0:
        print("numpy.linalg.linalg.LinAlgError: Singular matrix not reversible")
        return
    theta = theta.I * X.T * weighted * Y
    return theta

def lwlr_whole_dataset(X_shadow, X, Y, k):
    m, n = np.shape(X)
    predict = np.zeros( np.shape(Y) )
    for i in range(m):
        theta = locally_weighted_linear_regression(X_shadow[i], X, Y, k)
        predict[i] = X_shadow[i] * theta
    return predict

######################
def regression_error(Y, predict):
    return np.power(Y - predict, 2).sum()

def one_test_of_the_kernel(X, Y, k):
    predict = lwlr_whole_dataset(X[0:100], X[0:100], Y[0:100], k)
    error = regression_error(Y[0:100], predict)
    forecast_predict = lwlr_whole_dataset(X[100:200], X[0:100], Y[0:100], k)
    forecast_error = regression_error(Y[100:200], forecast_predict)
    print('k is {}, error {}, forecast error {}'.format(k, error, forecast_error))

def test_the_weighted_kernel(X, Y):
    """ As the value of k goes down, error goes down too, but forecast error goes up.
        Choose the right one.
    """
    values = [5.0, 2.0, 1.0, 0.7, 0.5, 0.2, 0.1, 0.077]
    for k in values:
        one_test_of_the_kernel(X, Y, k)

    theta = normal_equation(X[0:100], Y[0:100])
    print('normal equation error {}'.format(regression_error(Y[100:200], X[100:200] * theta)))


#####################
def sort_data_for_plot(X, Y, k):
    """ plot need sorted data
    """
    predict = lwlr_whole_dataset(X, X, Y, k)

    order = X[:, 1].argsort(0)
    X_order = X[order][:, 0, 1] # X[order] is 3d ( m x 1 x 1 )
    predict_order = predict[order][:, 0, :]
    return X_order, predict_order


def compare_draw_2d_lwlr(X, Y):
    """ x axes is X[:, 1], y axes is Y
        try weighted 1.0, 0.2, 0.05
        最小核 can minimum error，but have overfitting
    """
    fig = plt.figure()

    ax = fig.add_subplot(221)
    X_order, predict_order = sort_data_for_plot(X, Y, k=1.0)
    ax.scatter(X[:, 1].flatten().A[0], Y.flatten().A[0], s=2, c='red')
    ax.plot(X_order, predict_order, 'g')

    ax = fig.add_subplot(222)
    X_order, predict_order = sort_data_for_plot(X, Y, k=0.2)
    ax.scatter(X[:, 1].flatten().A[0], Y.flatten().A[0], s=2, c='red')
    ax.plot(X_order, predict_order, 'g')

    ax = fig.add_subplot(223)
    X_order, predict_order = sort_data_for_plot(X, Y, k=0.05)
    ax.scatter(X[:, 1].flatten().A[0], Y.flatten().A[0], s=2, c='red')
    ax.plot(X_order, predict_order, 'g')

    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Use to draw")
    parser.add_argument('--cost', '-c', help="plot cost function", action="store_true")
    parser.add_argument('--normal', '-n', help="calculate normal equation", action="store_true")
    parser.add_argument('--learn', '-l', help="plot one learning rate's convergence")
    parser.add_argument('--weighted', '-w', help="draw locally weighted linear regression", action="store_true")
    option = parser.parse_args()

    data, label = load_data('linear_data_3d.txt')
    data_norm, data_mean, data_std = feature_scaling(data)
    # initialize theta as zero
    theta = np.zeros((np.shape(data_norm)[1], 1))

    print(option.cost, option.normal, option.learn, option.weighted)
    if option.cost:
        plot_cost_function(data, label)
    if option.normal:
        theta_n = normal_equation(data, label)
        print('normal equation solve theta: {}\n'.format(theta_n))
    if option.learn:
        theta, cost_history = batch_gradient_descent(data_norm, label, theta)
        tune_learning_rate(cost_history)
        plot_result(data, label, theta, data_norm)
    if option.weighted:
        data, label = load_data('linear_data_2d.txt')
        test_the_weighted_kernel(data, label) 
        compare_draw_2d_lwlr(data, label)


if __name__ == '__main__':
    main()

