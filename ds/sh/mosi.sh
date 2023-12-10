#! /bin/bash
li=(v4.1.3_4gpu-80 v4.1.2_4gpu-10 v4.1.2_4gpu-20 v4.1.2_4gpu-30 v4.1.2_4gpu-40 v4.1.2_4gpu-50 v4.1-25 v4.1-50 v4.1-75 v4.1.5-25 v4.1.5-50 v4.1.5-75 v4.1.5-100 v4.3.2-25 v4.3.2-50 v4.3.2-75 v4.3.2-100 v4.3.4-25 v4.3.4-50 v4.1-100)
lsc=(no no no no no no no no no no no no no no no no no no no no)
for ((i=0;i<=19;i++))
do
python main.py --task mintrec --dont_show --output_file mintrec.csv --last_conv_layer ${lsc[$i]} --epochs 10 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file mintrec.csv --last_conv_layer ${lsc[$i]} --epochs 10 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file mintrec.csv --last_conv_layer ${lsc[$i]} --epochs 50 --patience 10 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file mintrec.csv --last_conv_layer ${lsc[$i]} --epochs 50 --patience 10 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mosi --dont_show --output_file mosi.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mosi --dont_show --output_file mosi.csv --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mosi --dont_show --output_file mosi.csv --last_conv_layer ${lsc[$i]} --epochs 50 --patience 10 --batch_size 24 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task mosi --dont_show --output_file mosi.csv --last_conv_layer ${lsc[$i]} --epochs 50 --patience 10 --batch_size 24 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
done
for ((i=0;i<=19;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap.csv --use_turn_ids --train_mode eleventurn --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --use_turn_ids --train_mode eleventurn --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --use_turn_ids --train_mode eleventurn --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --use_turn_ids --train_mode eleventurn --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
done
for ((i=13;i<=18;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap.csv --train_mode eleventurn --multi_audio --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --train_mode eleventurn --multi_audio --last_conv_layer ${lsc[$i]} --epochs 5 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --train_mode eleventurn --multi_audio --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 12 --accumulate_num 2 --lr 2e-5 --model saved_models/${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --train_mode eleventurn --multi_audio --last_conv_layer ${lsc[$i]} --epochs 50 --batch_size 12 --accumulate_num 1 --lr 2e-5 --model saved_models/${li[$i]}
done