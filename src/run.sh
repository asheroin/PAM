#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/merge_data_train.txt  \
 ./data_preprocess/merge_data_valid.txt \
 --gpu=2 \
 --save_dir=../output/bninception_trimmed_multi \
 --arch=bninception_trimmed_multi \
 --batch-size=128

