#!/usr/bin/env python
# coding=utf-8

import os


aes_path = 'train_train_list.txt'

aes_info = {'train':'train_train_list.txt',
        'valid':'train_valid_list.txt',
        'test':'test_list.txt'}

addb_info = {'train':'addb_Train.txt',
        'valid':'addb_Validation.txt',
        'test':'addb_Test.txt'}

def MakeSplit(split_mark):
    if split_mark not in ['train', 'valid', 'test']:
        raise Exception, 'Void Input'
    # aes
    aes_content = [x.strip() for x in open(aes_info[split_mark],'r').readlines()]
    addb_content = [x.strip() for x in open(addb_info[split_mark],'r').readlines()]
    with open('merge_data_{}.txt'.format(split_mark), 'w') as fp:
        for subitem in aes_content:
            img_path = subitem.split(' ')[0]
            img_score = subitem.split(' ')[1]
            fp.write(img_path)
            fp.write(' 0'*10)
            fp.write(' 1')
            fp.write(' 0'*10)
            fp.write(' {}\n'.format(img_score))
        for subitem in addb_content:
            subitems = subitem.split(',')
            fp.write(' '.join(subitems))
            fp.write('\n')

MakeSplit('train')
MakeSplit('valid')
MakeSplit('test')
