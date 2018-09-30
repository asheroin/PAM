#!/usr/bin/env python
# coding=utf-8


import os
import json
import numpy as np



# read results file
predictions = [float(x.strip().split(' ')[1]) for x in open('../evaluation_result.txt','r').readlines()]

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


