CUDA_VISIBLE_DEVICES=0 python /mnt/ewwe/yts/at/atppo/ppo/alignment_test.py \
    --batch_size=1 \
    --reward_path=/mnt/ewwe/yts/at/atppo/saved_models/v6.5.2-5 \
    --tokenizer_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/test.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/test_alignment.log 2>&1
