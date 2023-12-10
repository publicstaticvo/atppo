python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=1 \
    --lr=1e-4 --dont_show \
    --file_prefix /mnt/shared/yts \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.1.2 \
    --save_interval=2 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/shared/yts/spotify-960/transForTPP.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train612.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --file_prefix /mnt/shared/yts \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.0.3 \
    --save_interval=2 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/shared/yts/spotify-960/transForV4.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train503.log 2>&1
# python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/tpp/main.py \
#     --apex_level=2 \
#     --audio_length=10 \
#     --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
#     --batch_size=24 \
#     --grad_acc=1 \
#     --grad_norm=1 \
#     --lr=2e-5 --dont_show \
#     --file_prefix /mnt/shared/yts \
#     --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
#     --model_name=v6.1.1 \
#     --save_interval=10 \
#     --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
#     --train_epochs=50 \
#     --transcripts=/mnt/shared/yts/spotify-960/transForTPP.pkl
#     > /mnt/ewwe/yts/at/atppo/logs/train611.log 2>&1