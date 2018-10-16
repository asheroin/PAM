#!/usr/bin/env python
# coding=utf-8

import os
import random


for subidx in range(3):
    iterms = [x.strip() for x in open('train_list_{:03d}.txt'.format(subidx),'r').readlines()]
    length = len(iterms)

    print('total lens:{}'.format(length))

    random.shuffle(iterms)

    train_list = []
    valid_list = []

    for idx,val in enumerate(iterms):
        if idx < 0.9 * length:
            train_list.append(val)
        else:
            valid_list.append(val)

    train_list = sorted(train_list)
    valid_list = sorted(valid_list)

    fp_train = open('train_train_list_{:03d}.txt'.format(subidx),'w')
    fp_valid = open('train_valid_list_{:03d}.txt'.format(subidx),'w')

    fp_train.write('\n'.join(train_list))
    fp_train.close()

    fp_valid.write('\n'.join(valid_list))
    fp_valid.close()
