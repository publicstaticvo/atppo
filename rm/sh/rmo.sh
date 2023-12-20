python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=24 \
    --grad_acc=2 \
    --lr=1e-5 \
    --dont_show --perform_mlm \
    --num_negative=10 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5.1 \
    --save_interval=5 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transAnnotated.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train651.log 2>&1