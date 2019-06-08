#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

loss = gloss.L2Loss()
def log_rmse(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()

numeric_feature = train.dtypes[train.dtypes != 'object'].index
train[numeric_feature] = train[numeric_feature].apply(lambda x: (x - x.mean()) / (x.std()))

def massive_missing(df, threshod=0.6):
    nan_sum = df.isnull().sum()
    return nan_sum[nan_sum > df.shape[0] * threshod]

missing_series = massive_missing(train, 0.7)
train = train.drop(missing_series.index, axis=1)


def fill_missing_values(df):
    nan_sum = df.isnull().sum()
    nan_sum = nan_sum[nan_sum > 0]
    for column in list(nan_sum.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64':
            df[column].fillna(df[column].median(), inplace=True)
            
fill_missing_values(train)
train = pd.get_dummies(train)

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize()
    return net

def training(net, train_features, train_labels, num_epochs, learning_rate, batch_size):
    train_ls = []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        print('loss: {}, train rmse: {}'.format(l.mean().asscalar(), train_ls[-1]))
    return train_ls

train_features = nd.array(train.iloc[:, 1:-1].values)
train_labels = nd.array(train.SalePrice.values).reshape((-1, 1))

num_epochs, learning_rate, batch_size = 300, 0.1, 32
net = get_net()

train_l = training(net, train_features, train_labels, num_epochs, learning_rate, batch_size)
print(train_l)
print('avg train rmse %f' % (train_l[-1], ))

