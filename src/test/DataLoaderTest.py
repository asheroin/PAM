#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append('..')

from tqdm import tqdm

import utils.DataReader as DataReader
import utils.EpochRunner as EpochRunner


traindir = '/home/sujunjie/project/PAM/src/data_preprocess/train_list.txt'
train_loader = DataReader.GetTrainLoader(traindir, 512, 1)
batch_length = len(train_loader)
print('total length: {}'.format(batch_length))
for idx,(input,target) in tqdm(enumerate(train_loader)):
    pass
    # print('batch: {}/{}'.format(idx, batch_length))
    # print(input.shape)
    # print(target)


