#!/bin/bash

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/test_list.txt \
 --save_dir=../output/bninception \
 --arch=bninception \
 --batch-size=8 \
 --resume=../output/bninception/bninception_best.pth.tar \
 --evaluate

