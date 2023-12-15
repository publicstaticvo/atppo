#! /bin/bash

li=(v4.1_4GPU-10 v4.1_4GPU-20 v4.1_4GPU-30 v4.1_4GPU-40 v4.1_4GPU-50 v3.2.6-25 v3.2.7-25 v3.3.3-25)
lsc=(group group group group group group no group)
for ((i=0;i<=7;i++))
do
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file mosei.csv --epochs 5 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file mosei.csv --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file mosei.csv --epochs 50 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file mosei.csv --epochs 50 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
done