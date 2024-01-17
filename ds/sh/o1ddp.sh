#! /bin/bash

li=(v5.0.2-80 v5.0.2-100 v6.6-2 v6.6-4 v6.6-6 v6.6-8 v6.6-10 v6.1.2-20 v6.1.2-40 v6.1.2-60 v6.1.2-80 v6.1.2-100)
for ((i=0;i<=6;i++))
do
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
done
for ((i=0;i<=11;i++))
do
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file o1ddp.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
done
sleep 1m
for ((i=0;i<=11;i++))
do
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task mosei --dont_show --output_file ddp.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 50 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 5 --batch_size 24 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 main.py --task ic11 --dont_show --output_file ddp.csv --epochs 50 --batch_size 24 --accumulate_num 2 --lr 1e-5 --model ${li[$i]}
done