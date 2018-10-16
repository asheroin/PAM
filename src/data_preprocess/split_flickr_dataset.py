#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import random

# read json file
fp_str = open('../data_expro/user_score.json','r').read()
json_info = json.loads(fp_str)

test_user_name = []
train_user_name = []

test_images = []
total_images = []


# read raw labeled data
ROOT_PATH = '/home/sujunjie/project/PAM/dataset/FLICK-AES'
image_worker_pair_info = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_labeled_by_each_worker.csv'),'r').readlines()]
image_worker_pair = {}
for x in image_worker_pair_info[1:]:
    worker,image_idx,_ = x.split(',')
    if image_idx in image_worker_pair:
        image_worker_pair[image_idx].append(worker)
    else:
        image_worker_pair[image_idx] = [worker]

# read average scores
scores = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_score.txt'),'r').readlines()]
scores_dict = {x.split(' ')[0]:float(x.split(' ')[1]) for x in scores}

for key in json_info.keys():
    json_info[key] = [x for x in json_info[key] if len(image_worker_pair[x])>1 and x in scores_dict]

test_user_name = []
for key in json_info.keys():
    if len(json_info[key]) in range(80, 240):
        test_user_name.append(key)
    else:
        pass

print('{} users in the range'.format(len(test_user_name)))
sub_set_cnt = 0
total_tried = 0
select_set_list = []
while sub_set_cnt < 3 and total_tried < 15000:
    random.shuffle(test_user_name)
    selected_user = sorted (test_user_name[:37])
    total_images = []
    total_tried += 1
    # get total images number
    for user_key in selected_user:
        total_images += json_info[user_key]
    total_images = list(set(total_images))
    if len(total_images) in range(4500, 5200):
        print(selected_user)
        select_set_list.append(selected_user)
        print(len(total_images))
        sub_set_cnt += 1
    else:
        pass
        # print('fail to get a proper test set for the following users:\n{}'.format(selected_user_idx))
        # print('got image numbers = {}'.format(len(total_images)))
print('total tried time: {}'.format(total_tried))





# make train and test split
train_cnt = 0
IMAGE_ROOT = os.path.join(ROOT_PATH ,'40K')
# file in 40K
file_list = os.listdir(IMAGE_ROOT)


for sub_idx, select_set in enumerate(select_set_list):
    test_images_set = []
    for key in select_set:
        test_images_set += json_info[key]
    test_images_set = set(file_list)&set(test_images_set)
    train_images_set = set(file_list) - test_images_set
    print('train_len:{}'.format(len(train_images_set)))
    with open('train_list_{:03d}.txt'.format(sub_idx),'w') as fp:
        train_images_list = sorted(list(train_images_set))
        for x in train_images_list:
            if x not in scores_dict:
                continue
            if x in image_worker_pair and len(image_worker_pair[x]) <= 1:
                print('{} got {} scores'.format(x, len(image_worker_pair[x])))
                continue
            train_cnt += 1
            fp.write('{} {}\n'.format(os.path.join(IMAGE_ROOT, x), scores_dict[x]))


    print('test_len:{}'.format(len(test_images_set)))
    test_cnt = 0
    with open('test_list_{:03d}.txt'.format(sub_idx),'w') as fp:
        test_image_list = sorted(test_images_set)
        for x in test_image_list:
            if x not in scores_dict:
                continue
            test_cnt += 1
            fp.write('{} {}\n'.format(os.path.join(IMAGE_ROOT, x), scores_dict[x]))


