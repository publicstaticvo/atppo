#! /bin/bash

# li=(v6.5-1 v6.5-2 v6.5-3 v6.5-4 v6.5-5 v6.5-6 v6.5-7 v6.5-8 v6.5-9 v6.5-10 v6.5.1-1 v6.5.1-2 v6.5.1-3 v6.5.1-4 v6.5.1-5 v6.5.1-6 v6.5.1-7 v6.5.1-8 v6.5.1-9 v6.5.1-10)
li=(v6.1.3-10 v6.1.3-20 v6.1.3-30 v6.1.3-40 v6.1.3-50)
for ((i=0;i<=4;i++))
do
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model ${li[$i]}
done