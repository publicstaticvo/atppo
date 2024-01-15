python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --align_mode=word \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=10 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5.5 \
    --save_interval=2 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train655.log 2>&1