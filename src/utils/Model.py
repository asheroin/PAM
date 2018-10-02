#!/usr/bin/env python
# coding=utf-8

import sys
import os

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append('../ThirdParty/pretrained-models.pytorch/')

import pretrainedmodels


class ModelInterface(object):
    def __init__(self, arch, class_num = 1):
        # default class number is 1 for regrassion
        self.arch = arch
        if arch.startswith('bninception'):
            self.model = pretrainedmodels.models.bninception(num_classes = 1, pretrained = None)
        else:
            self.model = models.__dict__[arch](num_classes = class_num)
            if arch.startswith('inception'):
                # fix pytorch inception bugs for num_classes not equals 1000
                print('fixing inception models bugs')
                self.model.aux_logits = False
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, class_num)
    def GetModel(self):
        return self.model
    def SetEval(self):
        self.model.eval()
    def SetTrain(self):
        self.model.train()
    def ReadPretrain(self, model_path):
        model_path = os.path.join(model_path, self.arch + '.pth')
        checkpoint = torch.load(model_path, map_location = lambda storage, loc : storage)
        state_dict = {k: v for k,v in checkpoint.items() if 'fc' not in k}
        if self.arch.startswith('bninception'):
            state_dict= {k: v for k, v in checkpoint.items() if 'last_linear' not in k}
            for name, weights in state_dict.items():
                if '_bn.' in name:
                    state_dict[name] = weights.view(-1)
        model_dict = self.model.state_dict()
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict, strict = True)
        return 0
