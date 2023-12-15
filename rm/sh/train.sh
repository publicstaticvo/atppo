python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/rm/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=12 \
    --grad_acc=2 \
    --lr=1e-4 \
    --dont_show \
    --num_negative=10 \
    --model_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.1-80 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.3.1 \
    --save_interval=20 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=100 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForTPP.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train631.log 2>&1