python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/ppo/main.py \
    --actor_lr=1e-5 \
    --actor_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.2-20 \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=8 \
    --critic_lr=1e-5 \
    --critic_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.3.1-2 \
    --dont_show \
    --grad_acc=8 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5 \
    --reward_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.4-2 \
    --save_interval=1 \
    --tokenizer_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForPPO.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train650.log 2>&1
sleep 1m
python -m torch.distributed.launch --nproc_per_node=8 /mnt/ewwe/yts/at/atppo/ppo/main.py \
    --actor_lr=1e-5 \
    --actor_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.1.2-20 \
    --apex_level=2 \
    --audio_length=10 \
    --batch_size=8 \
    --critic_lr=1e-5 \
    --critic_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.3.1-6 \
    --dont_show \
    --grad_acc=8 \
    --model_save_path=/mnt/ewwe/yts/at/atppo/saved_models \
    --model_name=v6.5.1 \
    --reward_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.4-2 \
    --save_interval=1 \
    --tokenizer_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --train_epochs=10 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/transForPPO.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/train651.log 2>&1