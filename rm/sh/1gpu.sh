python /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=1 \
    --grad_acc=1 \
    --lr=1e-4 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.1-80 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/ \
    --model_name=test \
    --num_negative=10 \
    --save_interval=10 \
    --save_tmp=/mnt/ewwe/yts/at/atppo/models/debug \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=1 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/test.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/test.log 2>&1