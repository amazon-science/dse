PRETRAIN_DATA_DIR=$1

python main.py \
    --resdir /home/zhihan/data/dse_model \
    --datapath ${PRETRAIN_DATA_DIR} \
    --dataname tod_single_pos_3.tsv \
    --text sentence \
    --pairsimi pairsimi \
    --mode contrastive \
    --bert bertlarge \
    --contrast_type HardNeg \
    --lr 3e-06 \
    --lr_scale 100 \
    --batch_size 512 \
    --max_length 32 \
    --temperature 0.05 \
    --beta 1 \
    --epochs 15 \
    --max_iter 10000000 \
    --logging_step 400 \
    --feat_dim 128 \
    --num_turn 1 \
    --seed 1 \
    --save_model_every_epoch 
