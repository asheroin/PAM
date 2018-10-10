#!/bin/bash

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/merge_data_test.txt \
 --gpu=0 \
 --save_dir=../output/bninception_trimmed_multi \
 --arch=bninception_trimmed_multi \
 --batch-size=64 \
 --resume=../output/bninception_trimmed_multi/bninception_trimmed_multi_best.pth.tar \
 --evaluate

