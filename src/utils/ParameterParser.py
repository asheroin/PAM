#!/usr/bin/env python
# coding=utf-8

import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def GetParser():
    parser = argparse.ArgumentParser(description = 'default parser')
    parser.add_argument('data_train', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('data_val', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--save_dir', default='models', type=str, metavar='DIR',
                        help='direction to save model')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--gpu', default = 0, type=int,
                        help='No. of GPU devices')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0002, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--num_classes',default=14, type=int, help='num of class in the model')
    parser.add_argument('--dataset',default='places365',help='which dataset to train')


    return parser
