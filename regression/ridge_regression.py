#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>
#
# 特征数大于样本数，矩阵X不是满秩矩阵，逆不可求，岭回归解决。
# 缩减技术，需要对特征做标准化处理。
# 缩减系数，一些特征回归系数缩到0，减少模型复杂度。
# 通过交叉验证，得到最好缩减效果的alpha

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from linear_regression import load_data

def feature_normalize(X):
    """ `var` vs `std` both works
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[0, 0] = 1
    X_normalize = (X - X_mean) / X_std
    X_normalize[:, 0] = 1.0
    return X_normalize, X_mean, X_std

def ridge_regression(X, Y, alpha=0.2):
    """ If alpha is 0, the matrix can be singular
    """
    m, n = np.shape(X)
    theta = X.T * X + np.eye(n) * alpha
    if np.linalg.det(theta) == 0:
        print("numpy.linalg.linalg.LinAlgError: Singular matrix not reversible")
        return
    theta = theta.I * X.T * Y
    return theta

def generate_parameters(X, Y):
    m, n = np.shape(X)
    X_normalize, X_mean, X_std = feature_normalize(X)
    test_number = 60
    alphas = []
    thetas = np.zeros((test_number, n))
    for i, r in enumerate( range(test_number) ):
        alphas.append( np.exp(r-test_number/2.) )
        thetas[i, :] = ridge_regression(X_normalize, Y, alphas[-1]).T
    return thetas, alphas

def draw_parameters_trend(X, Y):
    """ If alpha is small, no parameters are shrink(the same as linear regression's).
        As alpha goes larger, all parameters shrink to zero.
        Conclusion: choose proper alpha.
    """
    ridge_weights, _ = generate_parameters(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(-30,30), ridge_weights) # plot of first dimension 和 alpha 定义域一致
    plt.show()

def cross_validation(X, Y):
    """ To get lowest error and best alpha
    """
    m, n = np.shape(X)
    ridge_weights, alphas = generate_parameters(X[:500], Y[:500])
    _, X_mean, X_std = feature_normalize(X[:500])
    X_test = (X[500:] - X_mean) / X_std
    for i, weight in enumerate(ridge_weights):
        # [:, np.newasix] change (3,) to (3,1)
        error = np.power(X_test * weight[:, np.newaxis] - Y[500:], 2).sum()
        print('alpha {}, test error is {}'.format(alphas[i], error))

def cross_validation_10(X, Y, num=10):
    """ 10折
        取最后一折的ridge_weights 来选缩减系数
    """
    m, n = np.shape(X)
    errors = np.zeros((num, 60))
    
    index_list = range(m)
    for i in range(num):
        np.random.shuffle(index_list)
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for j in range(m):
            if j < 0.9 * m:
                X_train.append(X[index_list[j]].A[0]) # (3,)
                Y_train.append(Y[index_list[j]].A[0])
            else:
                X_test.append(X[index_list[j]].A[0])
                Y_test.append(Y[index_list[j]].A[0])
        ridge_weights, alphas = generate_parameters(np.mat(X_train), np.mat(Y_train))
        X_normalize, X_mean, X_std = feature_normalize(np.mat(X_train))
        for k in range(len(ridge_weights)):
            _X = (np.mat(X_test) - X_mean) / X_std
            errors[i, k] = np.power(_X * ridge_weights[k][:, np.newaxis] - np.mat(Y_test), 2).sum()
    mean_error_of_alpha = np.mean(errors, 0)
    min_error_of_alpha  = np.min(mean_error_of_alpha)
    which_group_of_alpha = np.nonzero(mean_error_of_alpha == min_error_of_alpha)
    # print(ridge_weights.shape, errors.shape) # (60, 3), (10, 60)

    best_weight = ridge_weights[which_group_of_alpha]
    best_alpha  = np.mat(alphas).T[which_group_of_alpha[0]]
    print('best alpha {}, best weight {}'.format(best_alpha, best_weight))



if __name__ == '__main__':
    data, label = load_data('linear_data_3d.txt')
    draw_parameters_trend(data, label)
    cross_validation(data, label)
    cross_validation_10(data, label, num=10)

