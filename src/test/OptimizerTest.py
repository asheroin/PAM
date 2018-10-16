#!/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
from torchvision import models
sys.path.append('..')


import utils.ModelInterface as UtilsModel


model = UtilsModel.ModelInterface('bninception_trimmed_multi')





# base_params = [name for name,param in model.GetModel().state_dict().iteritems()]
# print(base_params)

base_lr = 0.001

model.GetModel().train()
for param in model.GetModel().parameters():
    param.requires_grad = True

inception_5b = [param for name, param in model.GetModel().state_dict().iteritems() \
                        if 'inception_5b_double_3x3_reduce' in name and 'bn' not in name]
inception_5b_name = [name for name, param in model.GetModel().state_dict().iteritems() \
                        if 'inception_5b' in name and 'bn' not in name]

print(len(inception_5b))
for elem in inception_5b_name:
    print elem

# fc layers
fc_params = [(name, param) for name, param in model.GetModel().state_dict().iteritems() \
                        if 'last_' in name or 'fc_' in name]
fc_weight_group = [param for name, param in fc_params if 'weight' in name]
fc_bias_group = [param for name, param in fc_params if 'bias' in name]
print(len(fc_weight_group))
print(len(fc_bias_group))
# build setting list for torch.optim
retlist = [
        {'params':inception_5b},
        # {'params':fc_weight_group, 'lr':10 * base_lr},
        # {'params':fc_bias_group, 'lr':20 * base_lr, 'weight_decay':0}
        ]



optimizer = torch.optim.SGD(retlist,
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 0.0002)


cnt = 0
print(len(optimizer.param_groups))
for param_group in optimizer.param_groups:
    print(param_group.keys())
    print(param_group['lr'])
    print(type(param_group['params']))
    print(len(param_group['params']))
    for idx,content in enumerate(param_group['params']):
        print('{} elements:'.format(idx))
        print(type(content))
        print(content)


