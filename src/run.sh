#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/train_valid_list.txt \
 --save_dir=../output/bninception \
 --arch=bninception \
 --batch-size=32

