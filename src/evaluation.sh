#!/bin/bash

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/test_list.txt \
 --save_dir=../output/inception_v3 \
 --arch=inception_v3 \
 --batch-size=8 \
 --resume=../output/inception_v3/inception_v3_best.pth.tar \
 --evaluate

