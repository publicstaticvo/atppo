#! /bin/bash

python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 5 --batch_size 20 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 5 --batch_size 20 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 50 --batch_size 20 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 50 --batch_size 20 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 5 --batch_size 20 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 5 --batch_size 20 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 50 --batch_size 20 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 50 --batch_size 20 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 5 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 50 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task mosei --dont_show --output_file mosei.csv --last_conv_layer group --epochs 50 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 5 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 5 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 50 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 50 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 5 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 5 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 50 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 50 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.3.3_1
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 5 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 50 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model saved_models/v3.2.6
python -m torch.distributed.launch --nproc_per_node=4 /root/workspace/mtt/main.py --system /root/data/yts --task meld --dont_show --output_file meld.csv --train_mode twentyturn --prompt --multi_audio --last_conv_layer group --epochs 50 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model saved_models/v3.2.6