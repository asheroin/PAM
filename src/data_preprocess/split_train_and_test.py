#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json


# read json file
fp_str = open('../data_expro/user_score.json','r').read()
json_info = json.loads(fp_str)

test_user_name = []
train_user_name = []

test_images = []
total_images = []

for key in json_info.keys():
    total_images.extend(json_info[key])
    if len(json_info[key]) in range(112, 181):
        test_user_name.append(key)
        test_images.extend(json_info[key])
    else:
        train_user_name.append(key)

test_images_set = set(test_images)
total_images_set = set(total_images)
train_images_set = total_images_set - test_images_set

print('total_images: {}'.format(len(total_images_set)))
print('train images: {}'.format(len(train_images_set)))
print('test images: {}'.format(len(test_images_set)))

# read average scores
ROOT_PATH = '/home/sujunjie/project/PAM/dataset/FLICK-AES'
scores = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_score.txt'),'r').readlines()]
scores_dict = {x.split(' ')[0]:float(x.split(' ')[1]) for x in scores}


# read raw labeled data
image_worker_pair_info = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_labeled_by_each_worker.csv'),'r').readlines()]
image_worker_pair = {}
for x in image_worker_pair_info[1:]:
    worker,image_idx,_ = x.split(',')
    if image_idx in image_worker_pair:
        image_worker_pair[image_idx].append(worker)
    else:
        image_worker_pair[image_idx] = [worker]


# make train and test split
train_cnt = 0
IMAGE_ROOT = os.path.join(ROOT_PATH ,'40K')
# file in 40K
file_list = os.listdir(IMAGE_ROOT)

with open('train_list.txt','w') as fp:
    train_images_list = sorted(list(set(file_list)&train_images_set))
    for x in train_images_list:
        if x not in scores_dict:
            continue
        if x in image_worker_pair and len(image_worker_pair[x]) <= 1:
            print('{} got {} scores'.format(x, len(image_worker_pair[x])))
            continue
        train_cnt += 1
        fp.write('{} {}\n'.format(os.path.join(IMAGE_ROOT, x), scores_dict[x]))


test_cnt = 0
with open('test_list.txt','w') as fp:
    test_image_list = sorted(list(set(file_list)&test_images_set))
    for x in test_image_list:
        if x not in scores_dict:
            continue
        test_cnt += 1
        fp.write('{} {}\n'.format(os.path.join(IMAGE_ROOT, x), scores_dict[x]))


print('real train:{}'.format(train_cnt))
print('real test:{}'.format(test_cnt))





