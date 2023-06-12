import torch
from utils import preprocess_word, pad_texts

from torch.utils import data as data_utils

import numpy as np
from config import pad_token, load_args
import pandas as pd

from scipy.io import loadmat

args = load_args()

class Batch:
	"Object for holding a batch of data with mask."
	def __init__(self, src, src_mask, trg, device='cuda'):
		## src : (B, D, T)
		self.src = src.transpose(1, 2).to(device) # (B, T, D)
		self.src_mask = src_mask.unsqueeze(-2).to(device) # (B, 1, T)
		self.src_lens = self.src_mask.long().sum((1, 2))
		self.trg = trg

		self.trg_lens = (self.trg != 0).long().sum(-1)

class Dataset:
	def __init__(self, args):
		with open(args.vocab_file) as f:
			self.chars = list(f.read().strip())

		blank_tokens = [pad_token]

		self.v = blank_tokens + self.chars
		self.v2i = {self.v[i]: i for i in range(len(self.v))}
		self.i2v = {i : v for v, i in self.v2i.items()}

		self.feat_root = args.feat_root

		test_df = pd.read_csv(args.test_csv, delimiter=',', dtype = str)
		self.fpaths = test_df["video_id"].tolist()
		if args.full_word_test:
			self.texts = test_df["corresponding_word"].tolist()
		else:
			self.texts = test_df["fingerspelled_letters"].tolist()
		self.starts = test_df["start"].tolist()
		self.ends = test_df["end"].tolist()

		self.upsampling = args.upsampling
		self.downsampling = args.downsampling
		self.frame_stride = args.frame_stride

	def __getitem__(self, idx):
		feat_path = self.fpaths[idx]
		start = int(float(self.starts[idx]) * 25./self.frame_stride)
		end = int(float(self.ends[idx]) * 25./self.frame_stride)
		text = self.texts[idx]

		feat_path = f"{args.feat_root}/{feat_path}/features.mat"

		feats = loadmat(feat_path)["preds"]

		ctc_target = torch.LongTensor(self.to_ids(text))
		feats = torch.FloatTensor(feats[start : end])

		if self.upsampling > 1:
			feats = feats.unsqueeze(1).repeat(1, self.upsampling, 1).reshape(-1, feats.size(-1))

		if self.downsampling > 1:
			feats = feats[::self.downsampling]

		return feats, ctc_target

	def collate_fn(self, batch):
		feats, ctc_targets = [], []

		for i in range(len(batch)):
			feats.append(batch[i][0])
			ctc_targets.append(batch[i][1])
			
		assert len(feats) != 0

		lens = np.array([len(f) for f in feats])
		max_len = max(lens)
		sorted_indices = np.argsort(-lens)
		feats = [feats[idx] for idx in sorted_indices]
		ctc_targets = [ctc_targets[idx] for idx in sorted_indices]

		padded_feats, src_mask = [], torch.zeros(len(feats), max_len)
		for i, f in enumerate(feats):
			if f.size(0) == max_len:
				padded_feats.append(f)
				src_mask[i] = 1
			else:
				padding = torch.zeros((max_len - f.size(0), f.size(1)))
				padded_feats.append(torch.cat([f, padding], dim=0))
				src_mask[i, :f.size(0)] = 1

		padded_ctc_targets, ctc_mask = pad_texts(ctc_targets)
		padded_feats = torch.stack(padded_feats, dim=0)
		if len(padded_feats.size()) == 3:
			padded_feats = padded_feats.transpose(1, 2) # (B, C, T)

		return padded_feats, src_mask, padded_ctc_targets.long()

	def __len__(self):
		return len(self.texts)

	def to_ids(self, word):
		return [self.v2i[c] for c in str(word)]

	def to_text(self, ids):
		try:
			return ''.join([self.i2v[i.item()] for i in ids if i != 0])
		except:
			return ''.join([self.i2v[i] for i in ids if i != 0])

def get_dataloader(args):
	dataset = Dataset(args)
	dataloader = data_utils.DataLoader(dataset, batch_size=args.bs, num_workers=args.num_workers, 
							collate_fn=dataset.collate_fn, 
							drop_last=False, shuffle=False)

	return dataloader
