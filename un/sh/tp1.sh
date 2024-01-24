python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=1 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=16 \
    --grad_acc=8 \
    --lr=1e-4 \
    --dont_show \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v7 \
    --save_interval=10 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=50 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transFor567.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train700.log 2>&1