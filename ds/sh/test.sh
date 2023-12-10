#! /bin/bash

python main.py --task mintrec --last_conv_layer layer --seed 42 --epochs 10 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/v2.2.2-125