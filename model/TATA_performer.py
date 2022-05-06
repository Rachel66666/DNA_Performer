import math
import logging
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

from performer_pytorch.performer_pytorch import PerformerLM, Performer, FastAttention, SelfAttention, CrossAttention, ProjectionUpdater
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from performer_pytorch.performer_enc_dec import PerformerEncDec



class DNA_Performer_Config:

    embd_pdrop = 0              # embedding dropout
    ff_dropout = 0            # feedforward dropout
    attn_dropout = 0

    n_embd= 1000
    block_size =100000
    n_head = 8
    n_encod = 5
    n_attention_layer= 6
    
    n_channel_cnn = 64
    n_intermediate = None
    max_short_seq_len=2000
    seq_len=100000

    causal = False
    feature_redraw_interval = 1000
    nb_features = None


    def __init__(self, n_input_val, n_output, seq_len,**kwargs):

        self.n_input_val = n_input_val
        self.n_output = n_output
        self.seq_len=seq_len
        for k,v in kwargs.items():
            setattr(self, k ,v)


class Conv_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        seq_len=config.seq_len

        self.encod = nn.Embedding(config.n_encod,config.n_encod)
        # config.n_encod is 5
        # config.n_embd is 1000
        self.cnn1 = nn.Conv1d(in_channels=config.n_encod, out_channels=64, kernel_size=8, stride=4, padding = 3)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=10, stride=5, padding = 4)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=config.n_embd, kernel_size=10, stride=5, padding = 4)
        

    def forward(self, x):
        
        x = self.encod(x)#.view(b,small_seq_len,w)
        b, _, small_seq_len, w = x.size()
        x = self.cnn1(x.view(b,small_seq_len,w).transpose(1, 2))
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        return x 


class Postion_Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_emb= nn.Parameter(torch.zeros(1, config.max_short_seq_len, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, tensor):
        b, seq_len,n_input_val = tensor.size()
        
        position_embeddings = self.pos_emb[:,:seq_len,:]
        tensor = self.drop(tensor + position_embeddings)
        
        return tensor


class Performer_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.performer_layers = Performer(
            dim = config.n_embd, 
            depth = config.n_attention_layer,
            heads = config.n_head,
            causal = config.causal,
            feature_redraw_interval = config.feature_redraw_interval,
            dim_head = config.n_embd//config.n_head,
            nb_features = config.nb_features,    
            attn_dropout = 0,
            ff_mult = 1
            )
    def forward(self, tensor):
        tensor = self.performer_layers(tensor)
        return tensor



# class PerformerLM_block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.performer_layers = PerformerLM(
#             dim = config.n_embd, 
#             depth = config.n_attention_layer,
#             heads = config.n_head,
#             causal = False,
#             num_tokens = 5,
#             max_seq_len = 100000
#             )
#     def forward(self, tensor):
#         tensor = self.performer_layers(tensor)
#         return tensor


class DNA_Performer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encod = nn.Embedding(config.n_encod,config.n_encod)
        self.acnn = Conv_Embedding(config)
        self.aembd = Postion_Embeddings(config)
        self.norm = nn.LayerNorm(config.n_embd)
        self.performer_layers = Performer_block(config)
        self.TATA_layer = nn.Linear(config.n_embd*1000, 2, bias=True)
        
        self.seq_len=config.seq_len

        
    def get_features(self, idx):
        
        
        #print("####before everything",idx.size())
        x = self.encod(idx)#.view(b,small_seq_len,w)
        #b, _, small_seq_len, w = x.size()
        #print("####after encod",x.size())
        x = self.acnn(idx)
        #print("####after cnn",x.size())
        x = self.aembd(x)
        #print("####after embd",x.size())
        
        x = self.performer_layers(x)
        #print("####after performer",x.size())
        
        x = self.norm(x)
        #print("####after norm",x.size())
        
        return x
    def forward(self, idx):
        x = self.get_features(idx)
        
        
        
        b, small_seq_len, w = x.size()
        x = x.view(b,small_seq_len*w)
        x = self.TATA_layer(x)
        
        
        #print("####output size",x.view(b,100000,4).size())
        return x


    