#! /bin/bash

li=(v6.2-10 old/v4.3.2-75)
for ((i=0;i<=9;i++))
do
python main.py --task mintrec --dont_show --output_file result.csv --epochs 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mintrec --dont_show --output_file result.csv --epochs 10 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 50 --patience 10 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosi --dont_show --output_file result.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosei --dont_show --output_file result.csv --epochs 5 --batch_size 32 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task mosei --dont_show --output_file result.csv --epochs 50 --batch_size 32 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
done

li=(v6.1-10 v6.1-20 v6.1-30 v6.1-40 v6.1-50 v6.1.1-20 v6.1.1-40 v6.1.1-60 v6.1.1-80 v6.1.1-100 v6.2.1-20 v6.2.1-40 v6.2.1-60 v6.2.1-80 v6.2.1-100 v6.2-10 old/v4.3.2-75)
for ((i=0;i<=14;i++))
do
python main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 5 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 5 --batch_size 16 --accumulate_num 1 --lr 2e-5 --model ${li[$i]}
python main.py --task iemocap --dont_show --output_file iemocap.csv --epochs 50 --batch_size 16 --accumulate_num 4 --lr 2e-5 --model ${li[$i]}
done