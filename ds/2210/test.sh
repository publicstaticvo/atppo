#! /bin/bash

CUDA_LAUNCH_BLOCKING=1 python main.py --task iemocap --train_mode eleventurn --use_turn_ids --last_conv_layer group --model saved_models/v3.3.1-100