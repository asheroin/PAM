#!/bin/bash

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/test_list.txt \
 --gpu=0 \
 --save_dir=../output/bninception_trimmed \
 --arch=bninception_trimmed \
 --batch-size=64 \
 --resume=../output/bninception_trimmed_Uint8Input_BGR/bninception_trimmed_best.pth.tar \
 --evaluate

