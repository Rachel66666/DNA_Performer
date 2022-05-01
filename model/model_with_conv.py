import math
import logging
import sys
from helpers import model_summary, model_summary2, cparam
import torch
import torch.nn as nn
from torch.nn import functional as F

#from performer_pytorch.performer_pytorch import PerformerLM, Performer, FastAttention, SelfAttention, CrossAttention, ProjectionUpdater
#from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
#from performer_pytorch.performer_enc_dec import PerformerEncDec



class DNA_Performer_Config:

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    n_embd= 3500
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


class Conv_block(nn.Module):
    def __init__(self, config):
        super().__init__()

        #seq_len=config.seq_len
        #self.encod = nn.Embedding(config.n_embd,config.n_embd)
        self.cnnb1 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd+1773, kernel_size=10, stride=1, padding = 4)
        self.cnnb2 = nn.Conv1d(in_channels=config.n_embd+1773, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
        '''
        self.cnnb3 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*3, kernel_size=10, stride=1, padding = 4)
        self.cnnb4 = nn.Conv1d(in_channels=config.n_embd*3, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
        self.cnnb5 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*2, kernel_size=10, stride=1, padding = 4)
        self.cnnb6 = nn.Conv1d(in_channels=config.n_embd*2, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
         self.cnnb7 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*2, kernel_size=10, stride=1, padding = 4)
        self.cnnb8 = nn.Conv1d(in_channels=config.n_embd*2, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
        self.cnnb9 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*2, kernel_size=10, stride=1, padding = 4)
        self.cnnb10 = nn.Conv1d(in_channels=config.n_embd*2, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
        self.cnnb11 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*2, kernel_size=10, stride=1, padding = 4)
        self.cnnb12 = nn.Conv1d(in_channels=config.n_embd*2, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
        self.cnnb13 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*2, kernel_size=10, stride=1, padding = 4)
        self.cnnb14 = nn.Conv1d(in_channels=config.n_embd*2, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)
        self.cnnb15 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd*2, kernel_size=11, stride=1, padding = 5)
        self.cnnb16 = nn.Conv1d(in_channels=config.n_embd*2, out_channels=config.n_embd, kernel_size=11, stride=1, padding = 5)
'''
    def forward(self, x):

        #x = self.encod(x)#.view(b,small_seq_len,w)
        #b, _, small_seq_len, w = x.size()
        #x = self.cnnb(x.view(b,small_seq_len,w).transpose(1, 2))
        x = x.transpose(1, 2)
        x = self.cnnb1(x)
        x = F.relu(x)
        x = self.cnnb2(x)
        x = F.relu(x)
        '''
        x = self.cnnb3(x)
        x = F.relu(x)
        x = self.cnnb4(x)
        x = F.relu(x)
        x = self.cnnb5(x)
        x = F.relu(x)
        x = self.cnnb6(x)
        x = F.relu(x)

        x = self.cnnb7(x)
        x = F.relu(x)
        x = self.cnnb8(x)
        x = F.relu(x)
        x = self.cnnb9(x)
        x = F.relu(x)
        x = self.cnnb10(x)
        x = F.relu(x)
        x = self.cnnb11(x)
        x = F.relu(x)
        x = self.cnnb12(x)
        x = F.relu(x)
        x = self.cnnb13(x)
        x = F.relu(x)
        x = self.cnnb14(x)
        x = F.relu(x)
        x = self.cnnb15(x)
        x = F.relu(x)
        x = self.cnnb16(x)
        x = F.relu(x)
        '''
        x = x.transpose(1, 2)

        return x




class DNA_Performer(nn.Module):

    def __init__(self, config):
        super().__init__()
        #self.encod = nn.Embedding(config.n_encod,config.n_encod)
        self.acnn = Conv_Embedding(config)
        self.aembd = Postion_Embeddings(config)
        self.norm = nn.LayerNorm(config.n_embd)
        self.conv_layers = Conv_block(config)
        self.expand_layer = nn.Linear(config.n_embd, 4*100, bias=True)

        self.seq_len=config.seq_len


        print("")
        print("Parameter Report")
        p_params = cparam(self.conv_layers)
        conv_params = cparam(self.acnn)
        posemb_params = cparam(self.aembd)
        #encod_params = cparam(self.encod)

        #print("Encode params:", encod_params)
        print("Conv params:", conv_params)
        print("Conv Layer params:", p_params)
        print("Positional Embedding params:", posemb_params)
        print("Total Params:", conv_params+p_params+posemb_params)

    def get_features(self, idx):


        print("####before everything",idx.size())
        #x = self.encod(idx)#.view(b,small_seq_len,w)
        #b, _, small_seq_len, w = x.size()
        #print("####after encod",x.size())
        x = self.acnn(idx)
        print("####after cnn",x.size())
        x = self.aembd(x)
        print("####after embd",x.size())

        x = self.conv_layers(x)
        print("####after Conv layer",x.size())



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
