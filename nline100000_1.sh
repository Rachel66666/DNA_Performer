#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 1

# Request 4 GPU
#$ -l gpus=2

# Request at least capability 3.5/6.0/7.0
#$ -l gpu_c=7.0

# Request at least 16 memory
#$ -l gpu_memory=16

#specify a project
#$ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m e

# Set maximum time to 20 days
#$ -l h_rt=600:00:00


module load python3/3.7.9
module load tensorflow/2.3.1
module load pytorch/1.7.0

lr=3e-4
batch_size=16
n_attention_layer=1
n_head=2
n_embd=3500
numlines=100000
seq_len=100000
epochs=200
lr_decay=1
emb_dropout=0
ff_dropout=0
attn_dropout=0


timestamp=$(date +%m-%d-%Y_%H-%M-%S)
logging_path_root=./runs2/
logging_path="${logging_path_root}embd_lr$lr/lrdecay$lr_decay/bs$batch_size/n_att_layer$n_attention_layer/head$n_head/n_embd$n_embd/seq_len$seq_len/em_dr$emb_dropout/ff_dr$ff_dropout/att_dr$attn_dropout/epoch$epochs/nlines$numlines/$timestamp"

#logging_path="./test"
mkdir -p $logging_path



python submit_training.py \
--logdir $logging_path \
--learning_rate $lr \
--lr_decay $lr_decay \
--batch_size $batch_size \
--n_attention_layer $n_attention_layer \
--n_head $n_head \
--n_embd $n_embd \
--numlines $numlines \
--seq_len $seq_len \
--epochs $epochs \
--emb_dropout $emb_dropout \
--ff_dropout $ff_dropout \
--attn_dropout $attn_dropout \