#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import operator
import numpy as np
import numpy.linalg as LA

from svd import svd

def euclid_distance(A, B):
    """ normalization of Euclid distance to (0, 1]

        return: 0 means A, B are very far, 1 means A == B
    """
    return 1. / (1 + LA.norm(A - B))

def pearson_correlation(A, B):
    """ not sensitive to magnitude, e.g. pearson([1,1]) == pearson([5,5])

        return: 0 means A, B are very far, 1 means A == B
    """
    if len(A) < 3: return 1
    return 0.5 + 0.5 * np.corrcoef(A, B, rowvar=0)

def cosine_similarity(A, B):
    """ If positive and negative numbers are in A or B, cosine(A, b) range in [-1, 1],
        0.5 + 0.5 * cosine(A, B) normalize result to [0, 1]
        Only positive number in A and B, cosine(A, B) range in [0.5, 1]

        return: 0 means A, B are very far, 1 means A == B
    """
    cosine = sum(A * B) / (LA.norm(A) * LA.norm(B))
    return 0.5 + 0.5 * cosine




def collaborative_filtering(X, user_row, distance=cosine_similarity, topN=5):
    """ item based collaborative filtering
        Assumption: row is user, column is product, score range in [0,5], 0 is not used

        遍历某用户所有未打分的物品，计算每个物品可能打分。
        对每个要计算打分的物品，遍历所有物品中用户打分过的物品，
        要计算打分的物品和打分过的物品，计算两者均有其他用户打分部分的相似度，
        用相似度乘以打分过的物品的打分之和，除以相似度之和(Why?)，得到估算的分数
    """
    m, n = np.shape(X)
    item_scores = []
    for unrated_col in np.nonzero(X[user_row, :].A == 0)[1]:
        count = 0; similarity = 0
        for j in range(n):
            user_rating = X[user_row, j]
            if user_rating == 0 or unrated_col == j: continue
            overlap = np.nonzero(np.logical_and(X[:, unrated_col].A > 0, X[:, j].A > 0))[0]
            if len(overlap) == 0: continue
            similarity += distance(X[overlap, unrated_col].flatten().A[0], X[overlap, j].flatten().A[0])
            count += similarity * user_rating
        estimate_score = similarity / count if count != 0 else 0
        item_scores.append((unrated_col, estimate_score))

    if item_scores == []: return None
    item_scores.sort(key=operator.itemgetter(1), reverse=True)
    return item_scores[:topN]


def cf(X, user_row, distance=cosine_similarity, topN=5):
    """ svd without overlap
        svd method do not need calculate overlap,
        because svd already have dimensionality reduction
    """
    m, n = np.shape(X)
    unrated_cols = np.nonzero(X[user_row, :].A == 0)[1]
    if len(unrated_cols) == 0: return None

    X_reduction = svd(X) # n*idx
    item_scores = []
    for unrated_col in unrated_cols:
        count = 0; similarity = 0
        for j in range(n):
            user_rating = X[user_row, j]
            if user_rating == 0 or unrated_col == j: continue
            similarity += distance(X_reduction[unrated_col, :].flatten().A[0], X_reduction[j, :].flatten().A[0])
            count += similarity * user_rating
        estimate_score = count / similarity if count != 0 else 0
        item_scores.append((unrated_col, estimate_score))
    item_scores.sort(key=operator.itemgetter(1), reverse=True)
    return item_scores[:topN]


if __name__ == '__main__':
    X = [[4, 4, 0, 2, 2],
         [4, 0, 0, 3, 3],
         [4, 0, 0, 1, 1],
         [1, 1, 1, 2, 0],
         [2, 2, 2, 0, 0],
         [1, 1, 1, 0, 0],
         [5, 5, 5, 0, 0]]

    r = collaborative_filtering(np.mat(X), 4)
    print(r)
    r = cf(np.mat(X), 4)
    print(r)

