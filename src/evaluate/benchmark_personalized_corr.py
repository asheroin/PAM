#!/usr/bin/env python
# coding=utf-8

import os
import sys
import scipy
from scipy.stats.stats import spearmanr
import numpy as np

open_info = """
This code is to produce a benchmark spearman rho when using average aesthics score as personlized score

usage: python THIS_FILE.py run
"""









def main():
    # read groundtruth
    gt_info = [x.strip() for x in open('../data_preprocess/test_list.txt','r').readlines()]

    gt_pic_list = [x.split(' ')[0] for x in gt_info]
    gt_pic_list = [x.split('/')[-1] for x in gt_pic_list]
    gt_pic_score = [float(x.split(' ')[1]) for x in gt_info]

    pred = {}
    for idx in range(len(gt_pic_list)):
        pred[gt_pic_list[idx]] = gt_pic_score[idx]

    # read user ranking
    user_info = [x.strip() for x in open('../../dataset/FLICK-AES/FLICKR-AES_image_labeled_by_each_worker.csv','r').readlines()]
    user_info = user_info[1:]

    user_json = {}

    for idx,val in enumerate(user_info):
        user_name, image_name, score = val.split(',')
        if image_name not in gt_pic_list:
            continue
        score = int(score)
        if user_name in user_json:
            user_json[user_name].append([image_name, score])
        else:
            user_json[user_name] = [[image_name, score]]

    user_json_for_metric = {}
    for key in user_json:
        user_json_for_metric[key] = []
        for val in user_json[key]:
            image_name, score = val
            user_json_for_metric[key].append([image_name, score, pred[image_name]])

    rho_list = []
    for user in user_json_for_metric:
        image_list, gt_score, pred_score =zip(*user_json_for_metric[user])
        gt_order = np.argsort(gt_score)[::-1]
        pred_order = np.argsort(pred_score)[::-1]
        # rho, pvalue = spearmanr(gt_order, pred_order)
        rho, pvalue = spearmanr(gt_score, pred_score)
        rho_list.append(rho)
        print('rho = {} for user {}'.format(rho, user))

    print('average rho = {}'.format(np.mean(rho_list)))







if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1]=='run':
        main()
    else:
        print(open_info)

