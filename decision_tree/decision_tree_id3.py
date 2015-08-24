#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

from collections import Counter
import numpy as np

import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__) + './../')
sys.path.append(parent_dir)
import toolkit

def source_entropy(Y):
    """ The result of entropy is [0, ∞)
    """
    m, _ = np.shape(Y)
    labels = Y.flatten().A[0]

    entropy = 0
    counter = Counter()
    for value in labels: counter[value] += 1

    for value, count in counter.iteritems():
        frequency = count / m
        entropy -= frequency * np.log2(frequency)

    return entropy

def choose_best_feature(data):
    """ Choose the feature which hava largest information gain,
    entropy of source tree minus this branch's source entropy.
    Which means we need to choose the smallest source entropy branch.

    The last column of data is Class.
    """
    m, n = np.shape(data)
    if n == 2: return 0
    mini_entropy = -1.
    best_feature = -1

    for i in range(n-1):
        counter = Counter()
        for value in data[:, i]: counter[value[0, 0]] += 1

        entropy = 0
        for val, times in counter.iteritems():
            branch_entropy = source_entropy( data[ (data[:, i].flatten().A[0] == val), -1 ] )
            entropy += times / n * branch_entropy

        if mini_entropy < 0 or entropy < mini_entropy:
            mini_entropy = entropy
            best_feature = i
    return best_feature


def make_tree(data, labels):
    """ If the tree only have one class, this branch over.
        If the tree only have class column, this branch over.
    """
    _, n = np.shape(data)
    if n == 1:
        counter = Counter()
        for value in data.flatten().A[0]: counter[value] += 1
        return counter.most_common(1)[0][0]
    if len( set(data[:, -1].flatten().A[0]) ) == 1:
        return data[0, -1]

    best_feature = choose_best_feature(data)
    tree = { labels[best_feature]: {} }

    for value in set(data[:, best_feature].flatten().A[0]):
        lines = range(best_feature) + range(best_feature+1, n)
        tree[labels[best_feature]][value] = \
            make_tree(data[ data[:, best_feature].flatten().A[0] == value ][:, lines],
                      labels[lines])
    return tree


def dump_tree(trees):
    import cPickle
    with open('trees.txt', 'w') as fd:
        cPickle.dump(trees, fd)

def load_tree():
    import cPickle
    with open('trees.txt') as fd:
        return cPickle.load(fd)



########  plot decision tree code

def get_width(tree):
    width = 0
    for label, value_tree in tree.iteritems():
        for value, blend in value_tree.iteritems():
            if isinstance(blend, dict):
                width += get_width(blend)
            else:
                width += 1
    return width

def get_height(tree):
    height = 1
    for label, value_tree in tree.iteritems():
        max_sub_height = 1
        for value, blend in value_tree.iteritems():
            if isinstance(blend, dict):
                sub_height = get_height(blend)
                if sub_height > max_sub_height:
                    max_sub_height = sub_height
    return height + max_sub_height

def draw_tree(tree):
    import matplotlib.pyplot as plt

    draw_tree.leaves = get_width(tree)
    draw_tree.layers = get_height(tree) - 1 # 3 layers means 2 line segment
    draw_tree.offset_x = - 0.5 / draw_tree.leaves
    draw_tree.offset_y = 1.

    def plot_this_arrow_text(parent_node, current_node, decision_text):
        middle_x = parent_node[0] + (current_node[0] - parent_node[0]) / 2.
        middle_y = parent_node[1] + (current_node[1] - parent_node[1]) / 2.
        draw_tree.axes.text(middle_x, middle_y, decision_text,
                            va='center', ha='center', rotation=35)

    def plot_this_node(node_text, parent_node, current_node, boxstyle):
        draw_tree.axes.annotate(node_text, xy=parent_node, xytext=current_node,
                                arrowprops={'arrowstyle': '<-'},
                                xycoords='axes fraction', textcoords='axes fraction',
                                va='center', ha='center',
                                bbox={'boxstyle': boxstyle, 'fc': '0.9'})

    def draw_recursively(tree, parent_node, decision_text):
        """ traversing a tree recursively, is depth first traversal from left to right.

        子树有N个叶子节点，则子树根节点应该放置的X轴位置为:
            current_position + .5 * N / total_leaf_number
        当前位置 + (总叶子节点数分之一的其中N份，再除以2取在左右中间)
        """
        width = get_width(tree)
        current_node = (draw_tree.offset_x + .5 * width / draw_tree.leaves, draw_tree.offset_y)

        plot_this_arrow_text(parent_node, current_node, decision_text)
        plot_this_node(tree.keys()[0], parent_node, current_node, 'round4')

        draw_tree.offset_y -= 1. / draw_tree.layers
        for decision_txt, sub_tree in tree.values()[0].iteritems():
            if isinstance(sub_tree, dict):
                draw_recursively(sub_tree, current_node, decision_txt)
            else:
                draw_tree.offset_x += 1. / draw_tree.leaves
                child_node = (draw_tree.offset_x, draw_tree.offset_y)
                plot_this_arrow_text(current_node, child_node, decision_txt)
                plot_this_node(sub_tree, current_node, child_node, 'circle')

        draw_tree.offset_y += 1. / draw_tree.layers

    fig = plt.figure(facecolor='white')
    fig.clf()
    draw_tree.axes = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    draw_recursively(tree, (.5, 1.), '')
    plt.show()

if __name__ == '__main__':

    # 16 samples have missing feature values, denoted by "?"
    data, _ = toolkit.load_data('../breast-cancer-wisconsin.data', label=False, sep=',', func=int)
    labels = np.array(['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'], copy=False)
    r = make_tree(data[:, 1:], labels[1:])
    print(r)

    r = make_tree(np.mat([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]), np.array(['no surfacing','flippers', 'Class']))
    print(r)
    draw_tree(r)

