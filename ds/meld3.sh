#! /bin/bash

lsc=(no no no no no no no no)
li=(v4.1.5-25 v4.1.5-50 v4.1.5-75 v4.1.5-100 v4.1.2_4gpu-40 v4.1.2_4gpu-50 v4.3.3-15 v4.3.5-25)
for ((i=0;i<=7;i++))
do
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]} --seed 3407
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei3.csv --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 24 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]} --seed 3407
done