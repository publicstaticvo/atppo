python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --num_ends=100 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.1 \
    --save_interval=10 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=50 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transFor567.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train510.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=2 \
    --lr=1e-4 \
    --dont_show \
    --num_ends=100 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v5.1.1 \
    --save_interval=20 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=100 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transFor567.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train511.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=1 \
    --lr=1e-4 \
    --dont_show \
    --num_ends=100 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.2 \
    --save_interval=10 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=50 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForTPP.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train620.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=4 /mnt/ewwe/yts/at/atppo/tpp/main.py \
    --apex_level=2 \
    --audio_length=10 \
    --audio_path=/mnt/ewwe/yts/at/models/wavlmForV1.1.3 \
    --batch_size=12 \
    --grad_acc=2 \
    --lr=1e-4 \
    --dont_show \
    --num_ends=100 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.2.1 \
    --save_interval=20 \
    --text_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=100 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForTPP.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train621.log 2>&1