#! /bin/bash

li=(v6.1.3-10 v6.1.3-20 v6.1.3-30 v6.1.3-40 v6.1.3-50)
for ((i=0;i<=4;i++))
do
python main.py --task mintrec --apex_level 1 --dont_show --output_file o1.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mintrec --apex_level 1 --dont_show --output_file o1.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python main.py --task mintrec --apex_level 1 --dont_show --output_file o1.csv --epochs 50 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mintrec --apex_level 1 --dont_show --output_file o1.csv --epochs 50 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 50 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --apex_level 1 --dont_show --output_file o1.csv --epochs 50 --batch_size 32 --accumulate_num 1 --lr 1e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file o1.csv --apex_level 1 --epochs 5 --batch_size 16 --accumulate_num 2 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file o1.csv --apex_level 1 --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file o1.csv --apex_level 1 --epochs 5 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model ${li[$i]}
python main.py --task ic11 --dont_show --output_file o1.csv --apex_level 1 --epochs 50 --batch_size 16 --accumulate_num 4 --lr 1e-5 --model ${li[$i]}
done