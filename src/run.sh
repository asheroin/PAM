#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/train_valid_list.txt \
 --save_dir=../output/resnet50 \
 --arch=resnet50 \
 --batch-size=64

