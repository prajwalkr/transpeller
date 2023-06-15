import numpy as np
import torch, os, copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import load_args
from models import builders

from dataloader import get_dataloader

args = load_args()
from dataloader import Batch

from utils import load, levenshtein

scaler = torch.cuda.amp.GradScaler()
from torch.cuda.amp import autocast

from fast_ctc_decode import beam_search, viterbi_search

def init(args):
	data_iter = get_dataloader(args)

	model = builders[args.builder](len(data_iter.dataset.v), args.feat_dim, 
							N=args.num_blocks, d_model=args.hidden_units, h=args.num_heads, 
							dropout=args.dropout_rate)

	return model.to(args.device), data_iter

def run_epoch(data_iter, model):
	total_cer, total_chars, total_samples = 0., 0, 0
	num_samples = len(data_iter.dataset)
	prog_bar = tqdm(data_iter)
	num_steps = 0

	for batch_idx, (feats, src_mask, trg) in enumerate(prog_bar):
		batch = Batch(feats, src_mask, trg, args.device)
		with autocast():
			binary_class_out, location_out, ctc_out = model(batch.src, 
															batch.src_mask)

		pred_texts = []
		gt_texts = [data_iter.dataset.to_text(ids[:l]) for ids, 
											l in zip(batch.trg, batch.trg_lens)]
		for idx, (c, l) in enumerate(zip(ctc_out, 
										batch.src_lens)):
			p, _ = beam_search(c[:l].cpu().numpy(), data_iter.dataset.v, 
								beam_size=args.beam)

			pred_texts.append(p)

		ids2text = np.vectorize(lambda x: data_iter.dataset.i2v[x])
		for i, (pred, gt) in enumerate(zip(pred_texts, gt_texts)):
			cer = levenshtein(list(gt), list(pred))
			total_cer += cer
			total_chars += len(gt)

		total_samples += len(feats)

		prog_bar.set_description('{} / {} | CER: {}'.format(total_samples, num_samples,
								round(total_cer / total_chars, 4)))

		if total_samples >= num_samples: return

def main(args):
	model, test_iter = init(args)

	assert args.ckpt_path is not None
	print('Resuming from: {}'.format(args.ckpt_path))
	model = load(model, args.ckpt_path)[0]

	with torch.no_grad():
		model.eval()
		run_epoch(test_iter, model)

if __name__ == '__main__':
	main(args)