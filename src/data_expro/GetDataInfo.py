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

print('Both in:{}'.format(len(set1&set2)))

# user information

user_information = [x.strip() for x in open(os.path.join(ROOT_PATH, 'FLICKR-AES_image_labeled_by_each_worker.csv'),'r').readlines()]
user_list = [x.split(',')[0] for x in user_information[1:]]
print('user set:{}'.format(len(set(user_list))))
labeled_list = list(set([x.split(',')[1] for x in user_information[1:]]))
print('labeled image:{}'.format(len(labeled_list)))

label_to_idx = {val:idx for idx,val in enumerate(labeled_list)}
labels_counter  = np.zeros([len(labeled_list)])

for x in user_information[1:]:
    x = x.split(',')[1]
    labels_counter[label_to_idx[x]] += 1

print(len(labels_counter))
print('max:{}'.format(np.max(labels_counter)))
print('min:{}'.format(np.min(labels_counter)))
print('mean:{}'.format(np.mean(labels_counter)))

for threshhold in [1,2,3,4,5,6]:
    filter_counts = 0
    for subitem in labels_counter:
        if subitem > threshhold:
            filter_counts += 1
    print('images with score more then {}: {}'.format(threshhold, filter_counts))

user_list = list(set(user_list))
user_to_idx = {val:idx for idx,val in enumerate(user_list)}
user_counter = np.zeros([len(user_to_idx)])

for x in user_information[1:]:
    x = x.split(',')[0]
    user_counter[user_to_idx[x]] += 1

print(len(user_counter))
print('max:{}'.format(np.max(user_counter)))
print('min:{}'.format(np.min(user_counter)))
print('mean:{}'.format(np.mean(user_counter)))

for threshhold in [50,100,200,300]:
    filter_counts = 0
    for subitem in user_counter:
        if subitem > threshhold:
            filter_counts += 1
    print('user with labeled image  more then {}: {}'.format(threshhold, filter_counts))


user_dict = {}

for subitem in user_information[1:]:
    username, picname, score = subitem.split(',')
    if username in user_dict:
        user_dict[username].append(picname)
    else:
        user_dict[username] = [picname]
json_str = json.dumps(user_dict, indent = 1)
with open('user_score.json','w') as fp:
    fp.write(json_str)

# to find the test set
possible_range = [[105,171], [105,172], [100,200],
                [100,181],[100,180],
                [112,181],[113,181]]

for test_range in possible_range:
    match_count = 0
    match_idxs = []
    match_sub_counts = []
    for idx, subitem in enumerate(user_counter):
        if subitem in range(test_range[0], test_range[1]):
            match_count += 1
            match_idxs.append(idx)
            match_sub_counts.append(subitem)
    print('from {} to {}(not include):'.format(test_range[0], test_range[1]))
    print('users maybe in testing set:{}'.format(match_count))
    print('average labeled images for possible user:{}'.format(np.mean(match_sub_counts)))
    # print images for test users
    test_images = []
    for user_idx in match_idxs:
        user_name = user_list[user_idx]
        user_images = user_dict[user_name]
        test_images.extend(user_images)
    test_images = list(set(test_images))
    print('all images for all users:{}'.format(len(test_images)))

























