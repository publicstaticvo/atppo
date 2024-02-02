#! /bin/bash

D=/mnt/ewwe/yts/at
# -m torch.distributed.launch --nproc_per_node=8
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-1 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-2 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-3 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-4 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-5 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-10 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-20 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-30 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-40 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
# python get_attention.py --apex_level=1 --model_path=$D/atppo/saved_models/v7-50 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-40 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-50 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-1 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-2 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-3 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-4 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-5 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-10 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-20 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
python get_attention.py --apex_level=1 --modality_reconstruct --model_path=$D/atppo/saved_models/v7.1-30 --model_save_path=$D/atppo/un/attn --text_path=$D/models/robertaForV3 --transcripts=$D/spotify-960/transFor567.pkl
