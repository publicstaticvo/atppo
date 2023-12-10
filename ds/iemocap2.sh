#! /bin/bash

li=(v4.3.2-25 v4.3.2-50 v4.3.2-75 v4.3.2-100 v4.3.4-25 v4.3.4-50 v4.3.4-75 v4.3.4-100)
for ((i=4;i<=7;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --use_turn_ids --epochs 5 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --use_turn_ids --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --use_turn_ids --epochs 50 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --use_turn_ids --epochs 50 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
done
for ((i=0;i<=7;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --multi_audio --epochs 5 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --multi_audio --epochs 5 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --multi_audio --epochs 50 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --multi_audio --epochs 50 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
done
for ((i=0;i<=7;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --use_turn_ids --epochs 5 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --use_turn_ids --epochs 5 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
done
for ((i=0;i<=7;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --multi_audio --epochs 5 --batch_size 12 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap_mtt.csv --train_mode eleventurn --multi_audio --epochs 5 --batch_size 12 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
done