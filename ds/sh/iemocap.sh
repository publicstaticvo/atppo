#! /bin/bash

li=(v6.1-10 v6.1-20 v6.1-30 v6.1-40 v6.1-50 v6.1.1-20 v6.1.1-40 v6.1.1-60 v6.1.1-80 v6.1.1-100 v6.2.1-20 v6.2.1-40 v6.2.1-60 v6.2.1-80 v6.2.1-100)
for ((i=0;i<=14;i++))
do
CUDA_VISIBLE_DEVICES=0 python main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
done