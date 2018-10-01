#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/train_valid_list.txt \
 --save_dir=../output/inception_v3 \
 --arch=inception_v3 \
 --batch-size=32

