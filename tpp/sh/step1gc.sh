deepspeed /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --ds_config=default \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=1 \
    --grad_acc=1 \
    --lr=1e-4 --ds_config=default \
    --grad_ckpt \
    --file_prefix /mnt/shared/yts \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.1.2 \
    --save_interval=2 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/shared/yts/spotify-960/transForTPP.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/test.log 2>&1
# python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
# deepspeed /mnt/ewwe/yts/at/atppo/tpp/main.py \  --ds_config=default \