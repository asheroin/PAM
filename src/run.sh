#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/train_valid_list.txt \
 --gpu=2 \
 --save_dir=../output/bninception \
 --arch=bninception \
 --batch-size=128

