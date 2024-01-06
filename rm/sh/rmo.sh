python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=20 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5.2 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train652.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=24 \
    --grad_acc=2 \
    --lr=1e-5 \
    --perform_mlm \
    --num_negative=20 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5.3 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train653.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --num_negative=20 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5.4 \
    --save_interval=1 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=5 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train654.log 2>&1