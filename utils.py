import os, torch, random
import numpy as np

import pickle 
from tqdm import tqdm
import re
import itertools

def load(model, ckpt_path, device='cuda'):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v

    model.load_state_dict(new_s)

    optimizer_state = checkpoint["optimizer"]
    
    epoch = checkpoint['global_epoch']

    if 'steps' in checkpoint:
        steps = checkpoint['steps']
        return model, optimizer_state, epoch, steps

    return model, optimizer_state, epoch, 0

def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was taken from: http://hetland.org/coding/python/levenshtein.py
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n
  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)
  return current[n]

def get_vocab(args):
    with open(args.vocab_file) as f: vocab = f.read()
    return vocab

def preprocess_word(word):
    word = word.lower()
    for c in ["'", "-"]: word = word.replace(c, '')
    return word

def pad_texts(texts):
    padded_texts = []
    text_mask = []
    max_len = max([len(t) for t in texts])
    for f in texts:
        if f.size(0) == max_len:
            padded_texts.append(f)
            text_mask.append(torch.ones(len(f)))
        else:
            padded_texts.append(torch.cat([f, torch.zeros(max_len - f.size(0))], dim=0))
            text_mask.append(torch.cat([torch.ones(len(f)), torch.zeros(max_len - len(f))]))

    padded_texts = torch.stack(padded_texts, dim=0)
    text_mask = torch.stack(text_mask, dim=0).long()

    return padded_texts, text_mask
