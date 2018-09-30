#!/bin/bash

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/test_list.txt \
 --save_dir=../output/resnet50 \
 --arch=resnet50 \
 --batch-size=8 \
 --resume=../output/resnet50/resnet50_best.pth.tar \
 --evaluate

