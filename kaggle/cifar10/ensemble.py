#!/usr/bin/env python
# -*- coding: utf-8 -*-

from train import *
from resnet import ResNet164_v2
from densenet import DenseNet

ctx = mx.gpu(0)

net1 = ResNet164_v2(10)
net1.load_parameters('./resnet.params', ctx=ctx)
net1.hybridize()

net2 = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
net2.load_parameters('./densenet.params', ctx=ctx)
net2.hybridize()


import pandas as pd

preds = []
for data, _ in test_data:
    data = data.as_in_context(ctx)
    output1 = nd.softmax(net1(data))
    output2 = nd.softmax(net2(data))
    output = 0.9523 * output1 + 0.9539 * output2
    pred_label = output.argmax(1)
    preds.extend(pred_label.astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_ds.synsets[x])
df.to_csv('ensemblesubmission.csv', index=False)
