#!/usr/bin/env python
# coding=utf-8

import torch
import os
import numpy as np

import utils.ParameterParser as ParameterParser
import utils.Model as UtilsModel
import utils.DataReader as DataReader
import utils.EpochRunner as EpochRunner

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    print('code start')
    args = ParameterParser.GetParser().parse_args()
    print('using following parameters:')
    print(args)
    # set to read default model
    # regrassion, set num to 1
    model = UtilsModel.ModelInterface(args.arch)
    model.ReadPretrain('../models/resnet50.pth')
    model.model = torch.nn.DataParallel(model.model, device_ids=[0]).cuda()
    print(model.model)
    # print model information
    if args.resume:
        if os.path.isfile(args.resume):
            # TODO: resume from model
            pass
        else:
            raise Exception("resume model not exits")
    # cudnn settings ???
    cudnn.benchmark = True
    # get data loader
    traindir = args.data_train
    validdir = args.data_val
    train_loader = DataReader.GetTrainLoader(traindir, args.batch_size, args.workers)
    valid_loader = DataReader.GetValidLoader(validdir, args.batch_size, args.workers)
    # cirterion
    criterion = nn.MSELoss().cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.GetModel().parameters(),
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay)

    if args.evaluate:
        EpochRunner.valid(valid_loader, model, criterion)
        return
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        # train a epoch
        EpochRunner.train(train_loader, model.GetModel(), criterion, optimizer, epoch)
        epoch_validation = EpochRunner.valid(valid_loader, model, criterion)
        is_best = epoch_validation > best_result
        best_resule = max(epoch_validation, best_result)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_value': best_result
            }, is_best, args.arch.lower())


def adjust_learning_rate(optimizer, epoch, argslr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = argslr * (0.1 ** (epoch // 30))
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    # lr = args.lr * (0.5 ** (epoch // 30))
    """Sets the learning rate more flexiable"""
    # if epoch<60:
    #     lr = args.lr
    # elif epoch<90:
    #     lr = args.lr * 0.1
    # else:
    #     lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    main()
