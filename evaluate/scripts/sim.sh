MODEL_DIR=$1
model_name=$2
DATA_DIR=$3
OUTPUT_DIR=$4


python run_similarity.py \
    --model_dir ${MODEL_DIR}/${model_name} \
    --data_root_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}/intent_sim/${model_name} \
    --TASK intent \
    --num_runs 10 \
    --max_seq_length 64

python run_similarity.py \
    --model_dir ${MODEL_DIR}/${model_name} \
    --data_root_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}/oos_sim/${model_name} \
    --TASK oos \
    --num_runs 10 \
    --max_seq_length 64


python run_similarity.py \
    --model_dir ${MODEL_DIR}/${model_name} \
    --data_root_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}/rs_ubuntu_sim/${model_name} \
    --TASK rs_ubuntu \
    --max_seq_length 128

python run_similarity.py \
    --model_dir ${MODEL_DIR}/${model_name} \
    --data_root_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}/rs_sim/${model_name} \
    --TASK rs_amazon \
    --max_seq_length 128
