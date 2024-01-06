#! /bin/bash

li=(v5-10 v5-20 v5-30 v5-40 v5-50 v5.0.1-20 v5.0.1-40 v5.0.1-60 v5.0.1-80 v5.0.1-100 v5.1.2-20 v5.1.2-40 v6-20 v6-40 v6-60 v6-80 v6-100 v6.0.1-20 v6.0.1-40 v6.0.1-60 v6.0.1-80 v6.0.1-100 v6.0.2-20 v6.0.2-40 v6.0.2-60 v6.0.2-80 v6.0.2-100 v6.0.3-20 v6.0.3-40 v5.0.2-20 v5.0.2-40 v5.0.2-60 v5.0.2-80 v5.0.2-100)
for ((i=0;i<=31;i++))
do
# python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
# python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
# python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
# python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
# python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 50 --batch_size 24 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
# python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 50 --batch_size 24 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
done