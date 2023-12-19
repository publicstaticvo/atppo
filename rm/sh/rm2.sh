python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --num_negative=10 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.2-20 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=20 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train640.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --num_negative=10 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.2.1-40 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.1 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=20 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForRM.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train641.log 2>&1