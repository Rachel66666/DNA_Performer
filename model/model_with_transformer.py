import math
import logging
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)
STATE_DICT_KEY_DEFAULT="state_dict"
CONFIG_KEY_DEFAULT="config"


class BERTConfig:
    """
    Configuration class for traditional transformer model
    """
    embd_pdrop = 0
    resid_pdrop = 0
    attn_pdrop = 0
    n_embd= 1024 #512#256#128#64 #768
    block_size =100000
    n_head = 8
    n_attention_layer = 6#7#8#12
    n_channel_cnn = 64
    n_intermediate = None
    max_short_seq_len=2000
    seq_len=100000
    n_encod = 5
    
    def __init__(self, n_input_val, n_output, seq_len,**kwargs):

        self.n_input_val = n_input_val
        self.n_output = n_output
        self.seq_len=seq_len
        for k,v in kwargs.items():
            setattr(self, k ,v)

class Conv_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encod = nn.Embedding(config.n_encod,config.n_encod)
        self.cnn1 = nn.Conv1d(in_channels=config.n_encod, out_channels=64, kernel_size=8, stride=4, padding = 3)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=10, stride=5, padding = 4)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=config.n_embd, kernel_size=10, stride=5, padding = 4)
        
    def forward(self, x):
        x = self.encod(x)
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
        b, seq_len,n_input_val = tensor.size() # t must be sequence length  # I doubt b is the bs
        position_embeddings = self.pos_emb[:,:seq_len,:]
        tensor = self.drop(tensor + position_embeddings)
        return tensor

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x,):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C //
                             self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # bi-directional self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)

        n_intermediate = config.n_intermediate
        if n_intermediate is None:
            n_intermediate = config.n_embd#4 *config.n_embd 

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, n_intermediate),
            nn.GELU(),
            nn.Linear(n_intermediate, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # Not using residual at all
        #print("before each self attention ", x.size())
        x = x + self.attn(self.ln1(x))
        #print("after each self attention ")
        x = x + self.mlp(self.ln2(x))
        #print("after each attention mlp")
        return x


class BERT(nn.Module):
    """
    Traditional transformer model
    """
    def __init__(self, config):
        super().__init__()
        self.acnn = Conv_Embedding(config)
        self.aembd = Postion_Embeddings(config)
        self.blocks = nn.Sequential(*[Block(config)
                      for _ in range(config.n_attention_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.seq_len=config.seq_len
        self.expand_layer = nn.Linear(config.n_embd, 4*100, bias=True)
        
    def get_features(self, idx):
        x = self.acnn(idx)
        x = self.aembd(x) # This is the position embeding layer
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, idx):
        features = self.get_features(idx)
        x = self.expand_layer(features)
        b, small_seq_len, w = x.size()
        return x.view(b,100000,4)


    def get_save_dict(self, config_key=CONFIG_KEY_DEFAULT, state_dict_key=STATE_DICT_KEY_DEFAULT):
        return {
            config_key: self.config,
            state_dict_key: self.state_dict()
        }

    def save_to_checkpoint(self, checkpoint_path, config_key=CONFIG_KEY_DEFAULT, state_dict_key=STATE_DICT_KEY_DEFAULT):
        torch.save(self.get_save_dict(config_key, state_dict_key), checkpoint_path)