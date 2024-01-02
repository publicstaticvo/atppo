python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=10 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.3.4 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train634.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=10 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.2.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.3 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train643.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=5 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.3.5 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train635.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=5 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.2.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.4 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train644.log 2>&1
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=10 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.2.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.3 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train643.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=20 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.3.6 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train636.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=20 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.2.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.5 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train645.log 2>&1