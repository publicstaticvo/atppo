#! /bin/bash
DIR=/mnt/ewwe/yts/at/atppo/ds

# CUDA_VISIBLE_DEVICES=0 python $DIR/main.py --task mintrec --batch_size 32 --epochs 10 --model old/v4.3.2-75 --output_file test.csv
# CUDA_VISIBLE_DEVICES=0 python $DIR/main.py --task mosi --batch_size 32 --model old/v4.3.2-75 --output_file test.csv
# CUDA_VISIBLE_DEVICES=0 python $DIR/main.py --task mosi --batch_size 32 --epochs 5 --model old/v4.3.2-75 --output_file test.csv
CUDA_VISIBLE_DEVICES=0 python $DIR/main.py --task ic11 --batch_size 16 --epochs 5 --model old/v4.3.2-75 --output_file test.csv --apex_level 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 $DIR/main.py --task iemocap --batch_size 24 --epochs 5 --model old/v4.3.2-75 --output_file test.csv
# CUDA_VISIBLE_DEVICES=0 python $DIR/main.py --task mosei --batch_size 24 --epochs 5 --model old/v4.3.2-75 --output_file test.csv
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --output_file test.csv --epochs 5 --batch_size 32 --accumulate_num 4 --model old/v4.3.2-75
