MODEL_DIR=$1
DATA_DIR=$2
OUTPUT_DIR=$3


# intent classification
for dataset in bank77 clinc150 hwu64 snips 
do  
    for data_ratio in 1 5
    do
        python run_finetune.py \
            --data_dir ${DATA_DIR}/intent/${dataset} \
            --model_type ${MODEL_DIR} \
            --TASK seq \
            --output_dir ${OUTPUT_DIR}/intent_ft/${MODEL_DIR}/${dataset}/${data_ratio} \
            --bert_lr 3e-5 \
            --epoch 50 \
            --max_seq_length 64 \
            --per_gpu_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --data_ratio ${data_ratio} \
            --num_runs 10 \
            --patience 5 \
            --classification_pooling average \
            --early_stop_type metric
    done
done

# response selection
for data_ratio in 500 1000 
do
    python run_finetune.py \
        --data_dir ${DATA_DIR}/rs/amazonqa \
        --model_type ${MODEL_DIR} \
        --TASK rs \
        --output_dir ${OUTPUT_DIR}/rs_ft/${MODEL_DIR}/amazonqa/${data_ratio} \
        --bert_lr 3e-5 \
        --epoch 50 \
        --max_seq_length 128 \
        --per_gpu_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --data_ratio ${data_ratio} \
        --num_runs 5 \
        --patience 3 \
        --eval_steps 50 \
        --concatenate
done

# dialogue action prediction
for dataset in dstc2 sim_joint
do
    for data_ratio in 10 20
    do
        python run_finetune.py \
            --data_dir ${DATA_DIR}/da/${dataset} \
            --model_type ${MODEL_DIR} \
            --TASK da \
            --output_dir ${OUTPUT_DIR}/da_concat_ft/${MODEL_DIR}/${dataset}/${data_ratio} \
            --bert_lr 5e-5 \
            --epoch 100 \
            --max_seq_length 32 \
            --per_gpu_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --data_ratio ${data_ratio} \
            --num_runs 5 \
            --patience 3 \
            --eval_steps 30 \
            --num_turn 1 \
            --concatenate \
            --save_model \
            --early_stop_type metric
    done
done
