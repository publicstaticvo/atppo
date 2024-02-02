import os
import sys
import tqdm
import pickle
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import *
from mlm_trainer import MaskedLMTrainer
from reconstruct_trainer import ReconstructTrainer
from dataset import ATDataset, UnsupervisedDataCollator
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaTokenizerFast

sw = None
SAMPLE_RATE = 16000
CONFIG = "config.json"
os.environ["NCCL_DEBUG"] = "WARN"

if __name__ == "__main__":
    # 1.输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--apex_level", default=0, type=int)
    parser.add_argument("--audio_length", default=10, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--file_prefix", default=None, type=str)
    parser.add_argument("--grad_acc", default=16, type=int)
    parser.add_argument("--grad_ckpt", action='store_true')
    parser.add_argument("--grad_norm", default=0., type=float)
    parser.add_argument("--modality_reconstruct", action='store_true')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_save_path", default=None, type=str)
    parser.add_argument("--num_sample_batch", default=1000, type=int)
    parser.add_argument("--num_turns", default=8, type=int)
    parser.add_argument("--text_length", default=512, type=int)
    parser.add_argument("--text_path", default=None, type=str)
    parser.add_argument("--transcripts", default=None, type=str, required=True)
    args = parser.parse_args()
    n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        args.apex_level = 0
    # 3。使用tokenizer
    config = ATConfig.from_pretrained(args.model_path)
    config.train_phase = 1
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)
    # 4。读输入数据
    train_data = ATDataset(args.transcripts, args.num_turns, args.file_prefix)
    c = UnsupervisedDataCollator(tokenizer, config, args.apex_level > 0, reconstruct=args.modality_reconstruct)
    # 5。整理config并建立模型
    model_class = ReconstructTrainer if args.modality_reconstruct else MaskedLMTrainer
    model = model_class.from_pretrained(args.model_path, config=config).to(args.device)
    # 6。数据并行
    if args.apex_level > 0:
        from apex import amp
        model = amp.initialize(model, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=c, sampler=SequentialSampler(train_data), num_workers=10)
    count = 0
    lengths = []
    output_path = f"{args.model_save_path}/{args.model_path.split('/')[-1]}"
    os.makedirs(output_path, exist_ok=True)
    for batch in train_loader:
        batch = to_device(batch, args.device)
        with torch.no_grad():
            attention, text_len = model(**batch, output_attentions=True)
        count += 1
        np.save(f"{output_path}/{count}.npy", attention[0].cpu().numpy())
        lengths.append(text_len)
        if count == args.num_sample_batch:
            break
    with open(f"{output_path}/lengths.pkl", "wb") as f:
        pickle.dump(lengths, f)
