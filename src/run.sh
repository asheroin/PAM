#!/bin/bash

# run train func

python main.py \
 ./data_preprocess/train_train_list.txt  \
 ./data_preprocess/train_valid_list.txt \
<<<<<<< HEAD
 --gpu=1 \
 --save_dir=../output/bninception \
 --arch=bninception \
=======
 --gpu=2 \
 --save_dir=../output/bninception_trimmed \
 --arch=bninception_trimmed \
>>>>>>> baseline/inception_trimmed
 --batch-size=128

