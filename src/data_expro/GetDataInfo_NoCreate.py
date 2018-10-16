#!/usr/bin/env python
# coding=utf-8


import sys
import os
import json

import numpy as np




ROOT_PATH = '/home/sujunjie/project/PAM/dataset/FLICK-AES/'


# filename set in score file
score_file_info = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_score.txt'),'r').readlines()]
file_name = [x.split(' ')[0] for x in score_file_info]

# image files in 40k
file_name_in_folder = os.listdir(os.path.join(ROOT_PATH, '40K'))

print('file_names in score file:{} ({})'.format(len(file_name), len(set(file_name))))

print('file_names in 40k:{} ({})'.format(len(file_name_in_folder),len(set(file_name_in_folder))))

set1 = set(file_name)
set2 = set(file_name_in_folder)
images_set = list(set1&set2)
print('Both in:{}'.format(len(set1&set2)))


# user information

user_information = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_labeled_by_each_worker.csv'),'r').readlines()]
user_list = [x.split(',')[0] for x in user_information[1:]]
print('user set:{}'.format(len(set(user_list))))
labeled_list = list(set([x.split(',')[1] for x in user_information[1:]]))
print('labeled image:{}'.format(len(labeled_list)))


## label #user

labeled_image_dict = {}
labeled_image_list = [x.split(',')[1] for x in user_information[1:]]
for image_name in labeled_image_list:
    if image_name in labeled_image_dict:
        labeled_image_dict[image_name] += 1
    else:
        labeled_image_dict[image_name] = 1



## user infomation
user_dict = {}
labeled_cnt = 0
less_scores = 0
for subitem in user_information[1:]:
    username, picname, score = subitem.split(',')
    if picname not in labeled_image_dict:
        # print('delete {} for having no label'.format(picname))
        labeled_cnt += 1
        continue
    if labeled_image_dict[picname] <= 2:
        # print('delete {} for having only {} scores'.format(picname, labeled_image_dict[picname]))
        less_scores += 1
        continue
    if username in user_dict:
        user_dict[username].append(picname)
    else:
        user_dict[username] = [picname]
print('{} images deleted for has no label recored for specific user'.format(labeled_cnt))
print('{} images deleted for little user giving score'.format(less_scores))



json_str = json.dumps(user_dict, indent = 1)

user_to_idx = {val:idx for idx,val in enumerate(user_list)}

# to find the test set
possible_range = [[105,171], [105,172], [100,200],
                [100,181],[100,180],
                [112,181],[113,181]]


for test_range in possible_range:
    match_count = 0
    match_idxs = []
    match_sub_counts = []
    real_test_image_list = []
    for key in user_dict:
        if len(user_dict[key]) in range(test_range[0], test_range[1]):
            match_count += 1
            match_idxs.append(key)
            match_sub_counts.append(len(user_dict[key]))
            real_test_image_list.extend(user_dict[key])
    real_test_image_list = list(set(real_test_image_list))
    print('from {} to {}(not include):'.format(test_range[0], test_range[1]))
    print('users maybe in testing set:{}'.format(match_count))
    print('average labeled images for possible user:{}'.format(np.mean(match_sub_counts)))
    print('total images: {}'.format(len(real_test_image_list)))








