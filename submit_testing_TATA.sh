module load python3/3.7.9
module load tensorflow/2.3.1
module load pytorch/1.7.0

logging_path_root=./testing_TATA/

logging_path="${logging_path_root}"
mkdir -p $logging_path

python testing_loop_TATA.py \
--logdir $logging_path \