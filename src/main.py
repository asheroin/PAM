#!/usr/bin/env python
# coding=utf-8

import os
import shutil
import time

import utils.ParameterParser as ParameterParser
import utils.ModelInterface as UtilsModel
import utils.DataReader as DataReader
import utils.EpochRunner as EpochRunner

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torchvision.models as models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('init {}'.format(m.__class__))
        # init.xavier_uniform(m.weight, gain = sqrt(2.0))
        init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        print('init {}'.format(m.__class__))
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)

def main():
    print('code start')
    args = ParameterParser.GetParser().parse_args()
    print('using following parameters:')
    print(args)
    # set to read default model
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    # regrassion, set num to 1
    model = UtilsModel.ModelInterface(args.arch)
    model.model.apply(weights_init)
    model.ReadPretrain('../models/')
    model.model = torch.nn.DataParallel(model.model, device_ids=[0]).cuda()
    print(model.model)
    # print model information
    best_result = 9999
    if args.resume:
        if os.path.isfile(args.resume):
            print('load model from file')
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_value']
            model.GetModel().load_state_dict(checkpoint['state_dict'])
            pass
        else:
            raise Exception("resume model not exits")
    # cudnn settings ???
    cudnn.benchmark = True
    # get data loader
    traindir = args.data_train
    validdir = args.data_val
    if 'multi' in args.arch:
        train_loader = DataReader.GetMultiTaskTrainLoader(traindir, args.batch_size, args.workers)
        valid_loader = DataReader.GetMultiTaskValidLoader(validdir, args.batch_size, args.workers)
    else:
        train_loader = DataReader.GetTrainLoader(traindir, args.batch_size, args.workers)
        valid_loader = DataReader.GetValidLoader(validdir, args.batch_size, args.workers)
    # cirterion
    criterion = nn.MSELoss(reduce=False).cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.GetModel().parameters(),
             lr = args.lr,
             momentum = args.momentum,
             weight_decay = args.weight_decay)
    # optimizer = torch.optim.Adam(model.GetModel().parameters(),
    #         lr = args.lr,
    #         weight_decay = args.weight_decay)
    if args.evaluate:
        print('evaluation...')
        loss_avg, target_list, output_list = EpochRunner.evaluate(valid_loader, model.GetModel(), criterion)
        print('instance number: {}'.format(len(target_list)))
        print(target_list[0])
        with open('evaluation_result.txt','w') as fp:
            for idx in range(len(target_list)):
                if 'multi' in args.arch:
                    fp.write('{} {}\n'.format(' '.join(map(str, target_list[idx]))
                        , ' '.join(map(str, output_list[idx]))))
                else:
                    fp.write('{} {}\n'.format(target_list[idx], output_list[idx]))
        return
    print('start training...')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.momentum)
        # train a epoch
        EpochRunner.train(train_loader, model.GetModel(), criterion, optimizer, epoch)
        with torch.no_grad():
            epoch_validation = EpochRunner.valid(valid_loader, model.GetModel(), criterion,epoch)
            print('validation#{}\tloss:{}'.format(epoch, epoch_validation))
        is_best = epoch_validation < best_result
        best_result = min(epoch_validation, best_result)
        save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.GetModel().state_dict(),
                'best_value': best_result
            }, is_best, args.save_dir, args.arch.lower())


def adjust_learning_rate(optimizer, epoch, argslr, argsmomentum):
    # decayed by 0.96 every epoch
    lr = argslr * (0.96 ** epoch)
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    # lr = args.lr * (0.5 ** (epoch // 30))
    """Sets the learning rate more flexiable"""
    # if epoch<60:
    #     lr = args.lr
    # elif epoch<90:
    #     lr = args.lr * 0.1
    # else:
    #     lr = args.lr * 0.01
    """
    https://discuss.pytorch.org/t/different-results-when-using-caffe-and-pytorch/3240/8
    different implement about m-SGD
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # fix momentum
        # param_group['momentum'] = argsmomentum / lr

def save_checkpoint(state, is_best, save_dir, filename = 'checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_dir, filename + '_latest.pth.tar'))
    if is_best:
        print('====> saving a best model@epoch {}\tloss = {}'.format(
            state['epoch'], state['best_value']
            ))
        shutil.copyfile(os.path.join(save_dir, filename + '_latest.pth.tar'),
                        os.path.join(save_dir, filename + '_best.pth.tar'))


if __name__=='__main__':
    main()
