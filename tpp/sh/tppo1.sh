python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=1 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=8 \
    --grad_acc=16 \
    --lr=5e-5 \
    --dont_show \
    --num_ends=100 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.1.3 \
    --save_interval=10 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=50 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transFor567.pkl \
    --weight_decay=0.0 \
    > /mnt/ewwe/yts/at/atppo/logs/train513.log 2>&1
