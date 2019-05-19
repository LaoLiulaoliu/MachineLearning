#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import datetime
import mxnet as mx
from mxnet import image
from mxnet import nd, gluon, autograd, init
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon import nn
from tensorboardX import SummaryWriter
import numpy as np
import shutil

def transform_train(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    im = nd.array(im, dtype='float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, rand_mirror=True,
                                    rand_crop=True,
                                   mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1)) # channel x width x height
    return im, nd.array([label]).astype('float32')

def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')

data_dir = '../data/kaggle_cifar10'
input_dir = 'train_valid_test'
train_ds = ImageFolderDataset(os.path.join(data_dir, input_dir, 'train'), transform=transform_train)
valid_ds = ImageFolderDataset(os.path.join(data_dir, input_dir, 'valid'), transform=transform_test)
train_valid_ds = ImageFolderDataset(os.path.join(data_dir, input_dir, 'train_valid'), transform=transform_train)
test_ds = ImageFolderDataset(os.path.join(data_dir, input_dir, 'test'), transform=transform_test)

train_data = DataLoader(train_ds, batch_size=64, shuffle=True, last_batch='keep')
valid_data = DataLoader(valid_ds, batch_size=64, shuffle=True, last_batch='keep')
train_valid_data = DataLoader(train_valid_ds, batch_size=64, shuffle=True, last_batch='keep')
test_data = DataLoader(test_ds, batch_size=128, shuffle=False, last_batch='keep')

criterion = gluon.loss.SoftmaxCrossEntropyLoss()

writer = SummaryWriter()

def get_acc(output, label):
    pred = output.argmax(1, keepdims=True)
    correct = (pred == label).sum()
    return correct.asscalar()

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_decay, epochs=[80, 140]):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        correct = 0
        total = 0
        if epoch in epochs:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            bs = data.shape[0]
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = criterion(output, label)
            loss.backward()
            trainer.step(bs)
            train_loss += nd.mean(loss).asscalar()
            correct += get_acc(output, label)
            total += bs
        writer.add_scalars('loss', {'train': train_loss / len(train_data)}, epoch)
        writer.add_scalars('acc', {'train': correct / total}, epoch)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_correct = 0
            valid_total = 0
            valid_loss = 0
            for data, label in valid_data:
                bs = data.shape[0]
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                output = net(data)
                loss = criterion(output, label)
                valid_loss += nd.mean(loss).asscalar()
                valid_correct += get_acc(output, label)
                valid_total += bs
            valid_acc = valid_correct / valid_total
            writer.add_scalars('loss', {'valid': valid_loss / len(valid_data)}, epoch)
            writer.add_scalars('acc', {'valid': valid_acc}, epoch)
            epoch_str = ("Epoch %d. Train Loss: %f, Train acc %f, Valid Loss: %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            correct / total, valid_loss / len(valid_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            correct / total))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))


def predict(model):
    import pandas as pd

    preds = []
    for data, _ in test_data:
        data = data.as_in_context(ctx)
        output = model(data)
        pred_label = output.argmax(1)
        preds.extend(pred_label.astype(int).asnumpy())

    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key = lambda x:str(x))

    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_ds.synsets[x])
    df.to_csv('submission.csv', index=False)
