export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_PORT=31415
export MASTER_ADDR=127.0.0.1 

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --n_epoch 20 \
        --student_type distilbert \
        --student_config training_configs/distilbert-base-multilingual-cased.json \
        --student_pretrained_weights serialization_dir/tf_bert-base-multilingual-cased_vocab_transformed.pth \
        --teacher_type bert \
        --teacher_name bert-base-multilingual-cased \
        --dump_path serialization_dir/my_first_training \
        --data_file data/binarized_text.bert-base-multilingual-cased.pickle \
        --token_counts data/token_counts.bert-base-multilingual_cased.pickle \
        --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --mlm