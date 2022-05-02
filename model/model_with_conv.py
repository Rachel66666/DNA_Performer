from helpers import model_summary, model_summary2, cparam
import torch
import torch.nn as nn
from torch.nn import functional as F



class DNA_Performer_Config:
    """
    Config class for CNN model
    """
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
        b, seq_len,n_input_val = tensor.size()
        position_embeddings = self.pos_emb[:,:seq_len,:]
        tensor = self.drop(tensor + position_embeddings)
        return tensor


class Conv_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnnb1 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd+1773, kernel_size=10, stride=1, padding = 4)
        self.cnnb2 = nn.Conv1d(in_channels=config.n_embd+1773, out_channels=config.n_embd, kernel_size=10, stride=1, padding = 5)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnnb1(x)
        x = F.relu(x)
        x = self.cnnb2(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        return x




class DNA_Performer(nn.Module):
    """
    CNN model
    """
    def __init__(self, config):
        super().__init__()
        self.acnn = Conv_Embedding(config)
        self.aembd = Postion_Embeddings(config)
        self.norm = nn.LayerNorm(config.n_embd)
        self.conv_layers = Conv_block(config)
        self.expand_layer = nn.Linear(config.n_embd, 4*100, bias=True)

        self.seq_len=config.seq_len

        # print("")
        # print("Parameter Report")
        # p_params = cparam(self.conv_layers)
        # conv_params = cparam(self.acnn)
        # posemb_params = cparam(self.aembd)
        # print("Conv params:", conv_params)
        # print("Conv Layer params:", p_params)
        # print("Positional Embedding params:", posemb_params)
        # print("Total Params:", conv_params+p_params+posemb_params)

    def get_features(self, idx):
        x = self.acnn(idx)
        x = self.aembd(x)
        x = self.conv_layers(x)
        x = self.norm(x)

        return x
    def forward(self, idx):
        features = self.get_features(idx)
        x = self.expand_layer(features)
        b, small_seq_len, w = x.size()
        return x.view(b,100000,4)