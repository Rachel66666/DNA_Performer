import math
import logging
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

from performer_pytorch.performer_pytorch import PerformerLM, Performer, FastAttention, SelfAttention, CrossAttention, ProjectionUpdater
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from performer_pytorch.performer_enc_dec import PerformerEncDec



class DNA_Performer_Config():

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    n_embd= 1024
    block_size =100000
    n_head = 8
    n_encod = 5
    
    n_channel_cnn = 64
    n_intermediate = None
    max_short_seq_len=2000
    seq_len=100000

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
        # config.n_encod is 5 (4(ATCG)+1(the masked one))
        # config.n_embd is 1000
        # The first parameter is num_embeddings (int) – size of the dictionary of embeddings
        # The second parameter is embedding_dim (int) – the size of each embedding vector
        # The n_encod is 5. 
        # Our volcab size is 5(4+1). So the first parameter is 5 = n_encod. 
        # For each number, we want a (5,1) vector to represent it. So the second parameter is 5 = n_encod. 
        self.cnn1 = nn.Conv1d(in_channels=config.n_encod, out_channels=64, kernel_size=8, stride=4, padding = 3)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=10, stride=5, padding = 4)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=config.n_embd, kernel_size=10, stride=5, padding = 4)
        
    def forward(self, x):
        
        x = self.encod(x)#.view(b,small_seq_len,w)
        print("after encod", x.size())
        b, _, small_seq_len, w = x.size()
        x = self.cnn1(x.view(b,small_seq_len,w).transpose(1, 2))
        print("after conv1", x.size())
        x = F.relu(x)
        x = self.cnn2(x)
        print("after conv2", x.size())
        x = F.relu(x)
        x = self.cnn3(x)
        print("after conv3", x.size())
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


class DNA_Performer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.acnn = Conv_Embedding(config)
        self.aembd = Postion_Embeddings(config)
        self.norm = nn.LayerNorm(config.n_embd)
        self.expand_layer = nn.Linear(config.n_embd, 4*100, bias=True)
        self.seq_len=config.seq_len

        
    def get_features(self, idx):
        
        print("####before everything",idx.size())
        x = self.acnn(idx)
        print("####after cnn",x.size())
        x = self.aembd(x)
        print("####after embd",x.size())
        
        x = self.norm(x)
        print("####after norm",x.size())
        
        return x
    def forward(self, idx):
        features = self.get_features(idx)
        
        x = self.expand_layer(features)
        print("####after linear",x.size())
        
        b, small_seq_len, w = x.size()
        
        
        print("####output size",x.view(b,100000,4).size())
        return x.view(b,100000,4)
