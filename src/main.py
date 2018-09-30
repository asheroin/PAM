#!/usr/bin/env python
# coding=utf-8

import torch
import os
import numpy as np

import utils.ParameterParser as ParameterParser
import utils.Model as UtilsModel
import utils.DataReader as DataReader
import utils.EpochRunner as EpochRunner

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models

def main():
    print('code start')
    args = ParameterParser.GetParser().parse_args()
    print('using following parameters:')
    print(args)
    # set to read default model
    # regrassion, set num to 1
    model = UtilsModel(args.arch)
    model.ReadPretrain('model_name')
    # print model information
    if args.resume:
        if os.path.isfile(args.resume):
            # TODO: resume from model
        else:
            raise Exception("resume model not exits")
    # cudnn settings ???
    cudnn.benchmark = True
    # get data loader
    train_loader = DataReader.GetTrainLoader(traindir, args.batch_size, args.workers)
    valid_loader = DataReader.GetValidLoader(validdir, args.batch_size, args.workers)
    # cirterion
    criterion = nn.MSELoss().cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay)

    if args.evaluate:
        EpochRunner.valid(valid_loader, model, criterion)
        return

    for epoch in range(args.start_eopch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train a epoch
        EpochRunner.train(train_loader, model, criterion, optimizer)
        epoch_validation = EpochRunner.valid(valid_loader, model, criterion)
        is_best = epoch_validation > best_result
        best_resule = max(epoch_validation, best_result)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_value': best_result
            }, is_best, args.arch.lower())


if __name__=='__main__':
    main()
