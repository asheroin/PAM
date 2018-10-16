#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import pandas as pd


ROOT_PATH = '/home/sujunjie/project/PAM/dataset/ADDB'
image_path = os.path.join(ROOT_PATH, 'datasetImages_warp256')
addb_mask = ','.join('1'*11)

def MakeSplit(split_mark):
    if split_mark not in ['Train', 'Test', 'Validation']:
        raise Exception,'Void Input'
    # get train file list and attribute name
    train_list = os.listdir(os.path.join(ROOT_PATH, 'imgListFiles_label'))
    train_list = [x for x in train_list if split_mark  in x and 'score' not in x and 'MotionBlur' not in x]
    # get attribute name
    attribute_name = [x.split('_')[1].split('.')[0] for x in train_list]
    attribute_name = sorted(list(set(attribute_name)))
    print(attribute_name)
    # aesthic scores
    file_path = os.path.join(ROOT_PATH, 'imgListFiles_label','imgList{}Regression_score.txt'.format(split_mark))
    file_content = [x.strip() for x in open(file_path, 'r').readlines()]

    addb_name = [x.split(' ')[0] for x in file_content]
    aes_score = [float(x.split(' ')[1]) for x in file_content]

    print('score range from {} to {}'.format(np.min(aes_score), np.max(aes_score)))

    with open('addb_{}.txt'.format(split_mark),'w') as fp_out:
        fps = []
        for subitem in attribute_name:
            subcont = open(os.path.join(ROOT_PATH, 'imgListFiles_label','imgList{}Regression_{}.txt'.format(split_mark,subitem))).readlines()
            sub_score = [float(x.split(' ')[1]) for x in subcont]
            fps.append(sub_score)
        for idx,val in enumerate(addb_name):
            fp_out.write(os.path.join(image_path, val))
            fp_out.write(','+addb_mask)
            for subidx,_ in enumerate(attribute_name):
                fp_out.write(',{}'.format(fps[subidx][idx]))
            fp_out.write(',{}\n'.format(aes_score[idx]))

MakeSplit('Train')
MakeSplit('Validation')
MakeSplit('Test')
