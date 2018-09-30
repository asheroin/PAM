#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/train_list_tiny.txt  \
 ./data_preprocess/test_list_tiny.txt \
 --save_dir=../output/resnet50 \
 --arch=resnet50 \
 --batch-size=32

