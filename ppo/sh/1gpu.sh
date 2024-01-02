python /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --actor_lr=2e-5 \
    --actor_path=/mnt/ewwe/yts/at/atppo/saved_models/v6-80 \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=1 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.6 \
    --reward_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.5-5 \
    --save_interval=100 \
    --tokenizer_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=1 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForPPO.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/test.log 2>&1