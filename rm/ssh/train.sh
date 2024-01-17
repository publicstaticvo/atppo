python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --align_mode=sent \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=4 \
    --dont_show \
    --grad_acc=8 \
    --lr=1e-5 \
    --num_negative=7 \
    --num_turns=2 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.2 \
    --save_interval=1 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=5 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train642.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --align_mode=sent \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=4 \
    --dont_show \
    --grad_acc=8 \
    --lr=1e-5 \
    --num_negative=7 \
    --num_turns=2 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.4.3 \
    --save_interval=1 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=5 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train643.log 2>&1