#! /bin/bash

li=(v6.1.2-20 v6.1.2-40 v6.1.2-60 v6.1.2-80 v6.1.2-100 v5.0.2-80 v5.0.2-100)
for ((i=0;i<=6;i++))
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