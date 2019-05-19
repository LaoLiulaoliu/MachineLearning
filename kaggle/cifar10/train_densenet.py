#!/usr/bin/env python
# -*- coding: utf-8 -*-

from train import *

from densenet import DenseNet
model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
model.initialize(ctx=mx.gpu(1))
model.hybridize()

ctx = mx.gpu(1)
num_epochs = 300
learning_rate = 0.1
weight_decay = 1e-4
lr_decay = 0.1
train(model, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx, lr_decay, [149, 224])

model.save_params('./densenet.params')

predict(model)
