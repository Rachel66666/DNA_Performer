import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


import model.model_with_performer as model
import model.model_with_transformer as Tmodel
import model.model_with_conv as Cmodel
import model.training_loop as trainer
import dataset.pickledataset as dataset

################################################################
import argparse

parser = argparse.ArgumentParser(description='Pretrain a BERT model on genomic data.')

parser.add_argument('--learning_rate', '-lr', type=float, default= 3e-4)
parser.add_argument('--batch_size', '-b', type=int, default=16)
parser.add_argument('--epochs', '-e', type=int, default=100000)

parser.set_defaults(lr_decay=False)
parser.add_argument('--n_attention_layer', type=int, default=12)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--numlines', type=int, default=992, help='number of lines of training data to process')
parser.add_argument('--lr_decay', type=int, default=0)
parser.add_argument('--emb_dropout', type=float, default=0)
parser.add_argument('--ff_dropout', type=float, default=0)
parser.add_argument('--attn_dropout', type=float, default=0)
parser.add_argument('--seq_len', type=int, default=100000, help='number of base pairs in one training sequences.')
parser.add_argument('--logdir', default='./runs/', help='Tensorboard logdir')
args = parser.parse_args()
################################################################




print("Pretraining Bert with the following arguments:")
for arg in vars(args):
    print(arg, getattr(args, arg))


# GPU setting: Define the device
device = 'cpu'# if no GPU then it is cpu 
if torch.cuda.is_available():
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device = torch.cuda.current_device() # if there is GPU then use GPU 

# Settings 

pickle_path = 'dataset/2000_line_100k_len_April_3.pickle'

lr = args.learning_rate
lr_decay= False
if args.lr_decay==0:
    lr_decay=False
else:
    lr_decay = True
batch_size=args.batch_size
n_attention_layer = args.n_attention_layer
n_head=args.n_head
seq_len=args.seq_len
epochs=args.epochs
numlines=(args.numlines-(args.numlines%args.batch_size)) #992
logdir=args.logdir
n_embd=args.n_embd
emb_dropout=args.emb_dropout
ff_dropout=args.ff_dropout
attn_dropout=args.attn_dropout



train_dataset = dataset.SeqDataset(pickle_path,numlines=numlines)


# Model part

# Performer
BERT = model.DNA_Performer
BERTConfig = model.DNA_Performer_Config

# Transformer
# BERT = Tmodel.BERT
# BERTConfig = Tmodel.BERTConfig

# CNN
# BERT = Cmodel.DNA_Performer
# BERTConfig = Cmodel.DNA_Performer_Config


# Creat instantiation
n_input_val = 1
n_output_val = 1
mconf = BERTConfig(n_input_val,n_output_val,seq_len=seq_len,n_attention_layer=n_attention_layer,n_head=n_head,n_embd=n_embd, attn_dropout=attn_dropout,ff_dropout=ff_dropout, embd_pdrop=emb_dropout)
model = BERT(mconf)


# Trainer part
Trainer = trainer.Trainer
TrainerConfig = trainer.TrainerConfig

# Creat instantiation
tconf = TrainerConfig(epochs=args.epochs, batch_size=batch_size,
                    learning_rate= lr,logdir= logdir,lr_decay = lr_decay,
                    model_name= logdir+'/model')

trainer = Trainer(model, train_dataset, tconf)

# Train
trainer.train()