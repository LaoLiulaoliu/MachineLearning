#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

import numpy as np
import matplotlib.pyplot as plt

def pca(X, topN=999):
    X_mean = np.mean(X, axis=0)
    X_anomaly = X - X_mean # 距平
    covariance = np.cov(X_anomaly, rowvar=0) # (n * n)
    eig_value, eig_vector = np.linalg.eig(np.mat(covariance)) # eig_vector (n * n)
    eig_value_idx = np.argsort(eig_value)
    top_eig_vector_idx = eig_value_idx[:-(topN+1):-1]
    top_eig_vector = eig_vector[:, top_eig_vector_idx]
    low_dimension = X_anomaly * top_eig_vector #(m * n) * (n * topN)
    refactor_X = low_dimension * top_eig_vector.T + X_mean
    return low_dimension, refactor_X


def variance_precentage(X):
    def plot_principle(y):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(y)), y, marker='^')
        plt.xlabel('Principal component number')
        plt.ylabel('percentage of variance')
        plt.show()

    X_mean = np.mean(X, axis=0)
    X_anomaly = X - X_mean # 距平
    covariance = np.cov(X_anomaly, rowvar=0) # (n * n)
    eig_value, eig_vector = np.linalg.eig(np.mat(covariance)) # eig_vector (n * n)
    sorted_eig_value = sorted(eig_value, reverse=True)
    variance_percentage = np.asarray(sorted_eig_value) / sum(sorted_eig_value) * 100
    plot_principle(variance_percentage)
    return variance_percentage


def plot_2_to_1(X, refactor_X):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0].flatten().A[0], X[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(refactor_X[:, 0].flatten().A[0], refactor_X[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

def replace_nan_with_mean(X):
    X = np.mat(X)
    _, n = np.shape(X)
    for i in range(n):
        mean_val = np.mean( X[np.nonzero( ~np.isnan(X[:, i]) )[0], i] )
        X[np.nonzero( np.isnan(X[:, i]) )[0], i] = mean_val
    return X

def test_replace_nan():
    X = np.ones((10, 3))
    X[1, 0] = np.nan
    X[7, 0] = 8
    X[5, 2] = np.nan
    X[7, 2] = 5
    replace_nan_with_mean(X)
    if X[1, 0] - 1.77777777778 < 1e10 and X[5, 2] - 1.44444444444 < 1e10:
        print('replace_nan_with_mean is right.')


if __name__ == '__main__':
    X = np.random.randn(30, 3)
    X = np.mat(X)
    low_dimension, refactor_X = pca(X)
    plot_2_to_1(X, refactor_X)

    variance_precentage(X)
