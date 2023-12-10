#! /bin/bash
li=(v4.3-25 v4.3-50 v4.3-75 v4.3-100 v4.3.2-25 v4.3.2-50 v4.3.2-75 v4.3.2-100 v4.3.4-25 v4.3.4-50 v4.3.4-75 v4.3.4-100)
for ((i=0;i<=11;i++))
do
python main.py --task mintrec --dont_show --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 2 --lr 1e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 1 --lr 1e-5 --model saved_models/${li[$i]}
done
for ((i=0;i<=11;i++))
do
python main.py --task mintrec --dont_show --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 2 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 1 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 2 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 1 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 2 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 10 --batch_size 16 --accumulate_num 1 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 2 --lr 5e-5 --model saved_models/${li[$i]}
python main.py --task mintrec --dont_show --seed 3407 --output_file small.csv --epochs 50 --patience 10 --batch_size 16 --accumulate_num 1 --lr 5e-5 --model saved_models/${li[$i]}
done