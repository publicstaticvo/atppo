#! /bin/bash
DIR=/mnt/ewwe/yts/at/atppo/ds

# python $DIR/main.py --task mintrec --batch_size 24 --model old/v4.3.2-75 --output_file test.csv
python $DIR/main.py --task iemocap --batch_size 24 --model old/v4.3.2-75 --output_file test.csv --epochs 5 --apex_level 1