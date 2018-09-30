#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import numpy as np

from scipy.stats.stats import spearmanr

results = json.loads(open('personalized_score.json','r').read())


pvalue_list = []
for user in results:
    print('user:{}'.format(user))
    image_list, gt_score, pred_score =zip(*results[user])
    gt_order = np.argsort(gt_score)[::-1]
    pred_order = np.argsort(pred_score)[::-1]
    rho, pvalue = spearmanr(gt_order, pred_order)
    pvalue_list.append(pvalue)
    print('p-value {}'.format(pvalue))

print('mean rank correlation: {}'.format(np.mean(pvalue_list)))

