#! /bin/bash

li=(v6.2.1-10 v6.2-20 v6.2-30 v6.2-40 v6.2-50)
for ((i=0;i<=1;i++))
do
CUDA_VISIBLE_DEVICES=0 python main.py --task mintrec --apex_level 1 --dont_show --output_file result2.csv --epochs 10 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task mintrec --apex_level 1 --dont_show --output_file result2.csv --epochs 10 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task mosi --apex_level 1 --dont_show --output_file result2.csv --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task mosi --apex_level 1 --dont_show --output_file result2.csv --epochs 5 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task mosi --apex_level 1 --dont_show --output_file result2.csv --epochs 50 --patience 10 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task mosi --apex_level 1 --dont_show --output_file result2.csv --epochs 50 --patience 10 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --apex_level 1 --task mosei --dont_show --output_file result2.csv --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --apex_level 1 --task mosei --dont_show --output_file result2.csv --epochs 50 --batch_size 24 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task iemocap --dont_show --output_file result2.csv --apex_level 1 --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0 python main.py --task iemocap --dont_show --output_file result2.csv --apex_level 1 --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --apex_level 1 --task iemocap --dont_show --output_file result2.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --apex_level 1 --task iemocap --dont_show --output_file result2.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
done