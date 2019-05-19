#!/usr/bin/env python
# -*- coding: utf-8 -*-

from train import *
from resnet import ResNet164_v2

model = ResNet164_v2(10)
model.initialize(ctx=mx.gpu(1), init=mx.initializer.Xavier())
model.hybridize()

ctx = mx.gpu(1)
num_epochs = 200
learning_rate = 0.1
weight_decay = 1e-4
lr_decay = 0.1
train(model, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx, lr_decay, [89, 139])

model.save_parameters('./resnet.params')

predict(model)
