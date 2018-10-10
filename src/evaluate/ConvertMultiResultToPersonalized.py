#!/usr/bin/env python
# coding=utf-8


import os
import json
import numpy as np


from scipy.stats.stats import spearmanr

# read results file
contents = [x.strip() for x in open('../evaluation_result.txt','r').readlines()]

attributes = contents[-1000:]
aesthics = contents[:-1000]




# attributes processing

# to get attribue name

ROOT_PATH = '/home/sujunjie/project/PAM/dataset/ADDB'
split_mark = 'Train'
# get train file list and attribute name
train_list = os.listdir(os.path.join(ROOT_PATH, 'imgListFiles_label'))
train_list = [x for x in train_list if split_mark  in x and 'score' not in x and 'MotionBlur' not in x]
# get attribute name
attribute_name = [x.split('_')[1].split('.')[0] for x in train_list]
attribute_name = sorted(list(set(attribute_name)))
print(attribute_name)

for attribute_idx in range(10):
    groundtruth = [float(x.split(' ')[attribute_idx]) for x in attributes]
    predictions = [float(x.split(' ')[11 + attribute_idx]) for x in attributes]
    resid = np.array(groundtruth) - np.array(predictions)
    loss = np.dot(resid, resid).mean()
    gt_order = np.argsort(groundtruth)
    pred_order = np.argsort(predictions)
    rho, pvalue = spearmanr(gt_order, pred_order)
    rho, pvalue = spearmanr(groundtruth, predictions)
    d_res = gt_order - pred_order
    d_res_sq = np.dot(d_res, d_res)
    N_ = len(groundtruth)
    pvalue = 1 - 6.0 * np.sum(d_res_sq) / (N_*N_ * N_ - N_)
    print('Attribute {}#{}: p-value = {}, rho = {}'.format(attribute_name[attribute_idx], attribute_idx, pvalue, rho))


predictions = [float(x.split(' ')[-1]) for x in aesthics]



# read groundtruth
gt_info = [x.strip() for x in open('../data_preprocess/test_list.txt','r').readlines()]

gt_pic_list = [x.split(' ')[0] for x in gt_info]
gt_pic_list = [x.split('/')[-1] for x in gt_pic_list]
gt_pic_score = [float(x.split(' ')[1]) for x in gt_info]

pred = {}
for idx in range(len(gt_pic_list)):
    pred[gt_pic_list[idx]] = predictions[idx]

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
json_str = json.dumps(user_json_for_metric, indent = 1)

with open('personalized_score.json','w') as fp:
    fp.write(json_str)

