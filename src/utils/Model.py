#!/usr/bin/env python
# coding=utf-8


import torch
import torchvision.models as models


class ModelInterface(object):
    def __init__(self, arch, class_num = 1):
        # default class number is 1 for regrassion
        self.model = models.__dict__[arch](num_classes = class_num)
    def GetModel(self):
        return self.model
    def SetEval(self):
        self.model.eval()
    def SetTrain(self):
        self.model.train()
    def ReadPretrain(self, model_path):
        checkpoint = torch.load(model_path, map_location = lambda storage, loc : storage)
        state_dict = {k: v for k,v in checkpoint.items() if 'fc' not in k}
        self.model.load_state_dict(state_dict, strict = False)
        return 0
