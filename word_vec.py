#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import math
import string
from collections import defaultdict

import numpy as np
import numpy.linalg as LA

class WordVec(object):
    def __init__(self, filename=None):
        if filename is None:
            filename = '../data/weibo_train_participle.txt'
        self.filename = filename

    def load_words(self):
        words = defaultdict(dict)
        Y = []
        with open(self.filename) as fd:
            for idx, line in enumerate(fd):
                item = line.split('\t', 6)
                words[idx] = filter(lambda x: x != '', map(string.strip, item[-1].split(',')))
                Y.append(map(int, (item[3], item[4], item[5])))
        return words, Y


    def bag_word(self, words, Y=None):
        """ m is the length of dataset
            n is length of wordset
            time complexity: m * length(word term)
        """
        vocabulary_dict = defaultdict(dict)
        for i, word_list in words.iteritems():
            for word in word_list:
                if word not in vocabulary_dict:
                    vocabulary_dict[word] = len(vocabulary_dict)

        word_set_len = len(vocabulary_dict)
        bag_words = []

        for i, word_list in words.iteritems():
            one_bag = [0] * word_set_len
            for word in word_list:
                one_bag[vocabulary_dict[word]] += 1
            if Y:
                one_bag.extend(Y[i])
            bag_words.append(one_bag)
        return bag_words, vocabulary_dict

    def build_lexicon(self, cropus):
        word_set = set()
        for i, word_list in cropus.iteritems():
            word_set.update(word_list)
        return word_set

    def bag_word_vec(self, words, vocabulary, Y=None):
        """ m is the length of dataset
            n is length of wordset
            time complexity: m * n * length(word term)
        """
        vocabulary = list(vocabulary)
        for i, word_list in words.iteritems():
            # term count
            one_bag = [word_list.count(word) for word in vocabulary]
            if Y:
                one_bag.extend(Y[i])
            yield one_bag


    def l2_normalizer(one_bag):
        """
        对每一个词频向量进行比例缩放，使它的每一个元素都在0到1之间，并且不会丢失太多有价值的信息。
        确保每个向量的L2范数等于1，一个计数为1的词在一个向量中的值和其在另一个向量中的值不再相同。
        如果想让一个文档看起来和一个特定主题更相关，你可能会通过不断重复同一个词，来增加它包含一个主题的可能性。
        在某种程度上，我们得到了一个在该词的信息价值上衰减的结果。所以我们需要按比例缩小那些在一篇文档中频繁出现的单词的值。
        """
        norm = LA.norm(np.asarray(one_bag))
        # norm = math.sqrt( sum([word**2 for word in one_bag]) )
        return [word / norm for word in one_bag]


    def inverse_doc_frequency(self, bag_words, vocabulary_dict):
        """ https://en.wikipedia.org/wiki/Tf%E2%80%93idf  Variants of TF weight
            the ratio inside the idf's log function is always greater than or equal to 1,
            the value of idf (and tf-idf) is greater than or equal to 0.
            the ratio inside the logarithm approaches 1, bringing the idf and tf-idf closer to 0.
        """
        sample_num = len(bag_words)
        def inverse_frequency(word):
            word_count = sum( [1 if word in one_bag else 0 for one_bag in bag_words] )
            return math.log( sample_num / (word_count + 1) )

        return [inverse_frequency(word) for word, idx in vocabulary_dict.iteritems()]

    def diagonal_idf_matrix(idf_vector):
        idf_len = len(idf_vector)
        idf_array = np.zeros((idf_len, idf_len))
        np.fill_diagonal(idf_array, idf_vector)
        return np.mat(idf_matrix)

    def tf_idf(self):
        words, Y = self.load_words()
        bag_words, vocabulary_dict = self.bag_word(words, Y)
        tf_matrix = np.mat(bag_words)
        idf_vector = self.inverse_doc_frequency(bag_words, vocabulary_dict)
        idf_matrix = self.diagonal_idf_matrix(idf_vector)
        tf_idf_matrix = tf_matrix * idf_matrix
        tf_idf_matrix_l2 = [l2_normalizer(vector.flatten().A[0]) for vector in tf_idf_matrix]
        return np.mat(tf_idf_matrix_l2)


class DataGen(object):
    def __init__(self, filename=None):
        self.wordvec = WordVec(filename)

    def generator_data(self, batch_num=10):
        words, Y = self.wordvec.load_words()
        vocabulary = self.wordvec.build_lexicon(words)
        one_bag = self.wordvec.bag_word_vec(words, vocabulary, Y)

        finish_flag = 0
        while finish_flag == 0:
            bags = []
            for i in range(batch_num):
                try:
                    one = one_bag.next()
                    bags.append(one)
                except:
                    finish_flag = 1
                    if bags != []: yield bags
                    break
            else:
                yield bags

    def explore_words(self):
        all_words = []
        words, Y = self.wordvec.load_words()
        for i, word_list in words.iteritems():
            for word in word_list:
                all_words.append(word)

        import pandas as pd
        df = pd.DataFrame(all_words)
        df.columns = ['word']
        top_50_words = df.word.value_counts()[:50]
        return top_50_words


if __name__ == '__main__':
    obj = DataGen('../data/weibo_train_participle.txt')
    bags = obj.generator_data()
    bags.next()

