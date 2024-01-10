#! /bin/bash

li=(v5.0.2-80 v5.0.2-100 v6.6-2 v6.6-4 v6.6-6 v6.6-8 v6.6-10)
for ((i=0;i<=6;i++))
do
python main.py --task mintrec --apex_level 1 --dont_show --output_file o1.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mintrec --apex_level 1 --dont_show --output_file o1.csv --epochs 10 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 5 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 50 --patience 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 50 --patience 10 --batch_size 32 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file o1.csv --apex_level 1 --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file o1.csv --apex_level 1 --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
done