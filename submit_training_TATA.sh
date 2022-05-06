#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 1

# Request 4 GPU
#$ -l gpus=1

# Request at least capability 3.5/6.0/7.0
#$ -l gpu_c=3.5

#$ -l gpu_memory=16

#specify a project
#$ -P dl523

#merge the error and output
# -j y

#send email at the end
#$ -m e

# Set maximum time to 20 days
#$ -l h_rt=300:00:00


module load python3/3.8.10
module load tensorflow/2.3.1
module load pytorch/1.7.0

lr=3e-4
batch_size=8
n_attention_layer=5
n_head=4
n_embd=3500
numlines=55000
seq_len=100000
epochs=20
lr_decay=1
emb_dropout=0
ff_dropout=0
attn_dropout=0
interval=100000000
nb_features=5


timestamp=$(date +%m-%d-%Y_%H-%M-%S)
logging_path_root=./runs_performer/
logging_path="${logging_path_root}TATA_load_performer/nb_features$nb_features/interval$interval/lr$lr/lrdecay$lr_decay/bs$batch_size/n_att_layer$n_attention_layer/head$n_head/n_embd$n_embd/seq_len$seq_len/att_dr$attn_dropout/epoch$epochs/nlines$numlines/$timestamp"



#logging_path="./test"
mkdir -p $logging_path



python submit_TATA_load_performer.py \
--logdir $logging_path \
--learning_rate $lr \
--lr_decay $lr_decay \
--batch_size $batch_size \
--n_attention_layer $n_attention_layer \
--n_head $n_head \
--n_embd $n_embd \
--interval $interval \
--numlines $numlines \
--seq_len $seq_len \
--epochs $epochs \
--emb_dropout $emb_dropout \
--ff_dropout $ff_dropout \
--attn_dropout $attn_dropout \
--nb_features $nb_features \