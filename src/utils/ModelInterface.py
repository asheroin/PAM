#!/usr/bin/env python
# coding=utf-8


import CustomModel as custom_model

import sys
import os

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append('../ThirdParty/pretrained-models.pytorch/')

import pretrainedmodels


def TermsNotInK(lists, k):
    for term in lists:
        if term in k:
            return False
    return True

class ModelInterface(object):
    def __init__(self, arch, class_num = 1):
        # default class number is 1 for regrassion
        self.arch = arch
        if arch == 'bninception':
            self.model = pretrainedmodels.models.bninception(num_classes = class_num, pretrained = None)
        elif arch == 'bninception_trimmed':
            self.model = custom_model.bninception_trimmed(num_classes = class_num, pretrained = None)
        elif arch == 'bninception_trimmed_multi':
            self.model = custom_model.bninception_trimmed_multi(num_classes = class_num, pretrained = None)
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
        if 'trimmed' in self.arch:
            # must be bninception serise
            print('[DEBUG] reading checkpoint from bninception.pth')
            model_path = os.path.join(model_path, 'bninception.pth')
        else:
            model_path = os.path.join(model_path, self.arch + '.pth')
        checkpoint = torch.load(model_path, map_location = lambda storage, loc : storage)
        state_dict = {k: v for k,v in checkpoint.items() if 'fc' not in k}
        if self.arch.startswith('bninception'):
            if 'trimmed' in self.arch:
                state_dict= {k: v for k, v in checkpoint.items() if TermsNotInK(['last_linear','inception_5b_1x1','inception_5b_double_3x3_2','inception_5b_pool_proj'] ,k)}
            else:
                state_dict= {k: v for k, v in checkpoint.items() if TermsNotInK(['last_linear'] ,k)}
            for name, weights in state_dict.items():
                if '_bn.' in name:
                    state_dict[name] = weights.view(-1)
        model_dict = self.model.state_dict()
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict, strict = True)
        return 0


