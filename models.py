import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LocalizerEncoderOnlyCTC, \
        PositionwiseFeedForward, PositionalEncoding, EncoderLayer, \
        MultiHeadedAttention, Encoder, Generator

def LocalizerCTC(vocab, visual_dim=1024, N=6, d_model=512, h=8, dropout=0.1):
    c = copy.deepcopy

    d_ff = 4 * d_model

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = LocalizerEncoderOnlyCTC(
        in_proj=nn.Sequential(nn.Linear(visual_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), 
                                nn.Linear(d_model, d_model)),
        video_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        pos_embed=c(position), 
        classifier=nn.Linear(d_model, 1), detector=nn.Linear(d_model, 1), 
        recognizer=Generator(d_model, vocab),
        cls_tag=nn.Parameter(torch.randn(1, 1, d_model)))
    
    # Initialize parameters with Glorot / fan_avg.
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model



builders = {
            'localizer_ctc': LocalizerCTC,
        }