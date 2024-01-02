python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --actor_lr=2e-5 \
    --actor_path=/mnt/ewwe/yts/at/atppo/saved_models/v6-80 \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=8 \
    --dont_show \
    --grad_acc=4 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.6 \
    --reward_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.5-5 \
    --save_interval=10 \
    --tokenizer_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=50 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForPPO.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train660.log 2>&1