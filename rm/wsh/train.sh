python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --align_mode=word \
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