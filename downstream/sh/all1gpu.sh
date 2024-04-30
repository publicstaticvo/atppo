#! /bin/bash

li=(v6.5-1 v6.5-2 v6.5-3 v6.5-4 v6.5-5 v6.5-6 v6.5-7 v6.5-8 v6.5-9 v6.5-10 v6.5.1-1 v6.5.1-2 v6.5.1-3 v6.5.1-4 v6.5.1-5 v6.5.1-6 v6.5.1-7 v6.5.1-8 v6.5.1-9 v6.5.1-10)
for ((i=0;i<=19;i++))
do
python main.py --task mintrec --dont_show --output_file result.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mintrec --dont_show --output_file result.csv --epochs 10 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 50 --patience 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file iemocap.csv --epochs 5 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file iemocap.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file iemocap.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
done