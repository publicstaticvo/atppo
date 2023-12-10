python /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=8 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --file_prefix /mnt/shared/yts \
    --model_save_path=/mnt/ewwe/yts/at/atppo/ \
    --model_name=test \
    --save_interval=10 \
    --save_tmp=/mnt/ewwe/yts/at/atppo/models/debug \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=1 \
    --transcripts=/mnt/shared/yts/spotify-960/test.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/test.log 2>&1