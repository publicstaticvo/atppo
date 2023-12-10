python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=48 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --file_prefix /mnt/shared/yts \
    --grad_norm=1 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.0.1 \
    --save_interval=2 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/shared/yts/spotify-960/transForV4.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train501.log 2>&1
#     --save_tmp=/mnt/ewwe/yts/at/atppo/models/debug \
sleep 1m
python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=48 \
    --grad_acc=1 \
    --lr=2e-5 \
    --dont_show \
    --file_prefix /mnt/shared/yts \
    --grad_norm=1 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.0.2 \
    --save_interval=10 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=50 \
    --transcripts=/mnt/shared/yts/spotify-960/transForV4.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train502.log 2>&1