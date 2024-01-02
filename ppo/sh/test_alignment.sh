python /mnt/ewwe/yts/at/atppo/ppo/alignment_test.py \
    --batch_size=1 \
    --tokenizer_path=/mnt/ewwe/yts/at/models/robertaForV3 \
    --transcripts=/mnt/ewwe/yts/at/spotify-960/test.pkl \
    > /mnt/ewwe/yts/at/atppo/logs/test_alignment.log 2>&1
