python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=4 \
    --lr=1e-4 \
    --dont_show \
    --grad_norm=1 \
    --num_ends=100 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.1.2 \
    --save_interval=20 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=100 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transFor567.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train512.log 2>&1
