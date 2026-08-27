#!/bin/bash
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "Please set CUDA_VISIBLE_DEVICES before running this script."
    echo "Example: CUDA_VISIBLE_DEVICES=0 bash $0"
    exit 1
fi

lm_name='allenai/longformer-large-4096'
simple_lm_name='longformer_large'
opt=3
max_seq_len=1024
global=4
win_size=256
model_type='classification'
scoring='softmax'
lr=1e-5
dropout_rate=0.1
epochs=3
max_patience=3
GA=1
optimizer='adamw'
warmup_proportion=0.1
batch_size=8
eval_batch_size=64
seed=1013
mask_mention_prob=0.0

data_path=./data/stage1/instances_longformer_tok_opt${opt}_seq_len${max_seq_len}_seed${seed}
save_path=./save_stage1/${model_type}_${scoring}_${simple_lm_name}/win${win_size}/mask_mention_prob${mask_mention_prob}/dropout${dropout_rate}/opt${opt}_seq_len${max_seq_len}_epochs${epochs}_global${global}_lr${lr}_win${win_size}
python3 main.py \
    --data_path ${data_path} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --lm_name ${lm_name} \
    --attention_window_size ${win_size} \
    --global_mode ${global} \
    --model_type ${model_type} \
    --scoring ${scoring} \
    --mask_mention \
    --mask_mention_prob ${mask_mention_prob} \
    --dropout_rate ${dropout_rate} \
    --optimizer ${optimizer} \
    --lr ${lr} \
    --warmup_proportion ${warmup_proportion} \
    --weight_decay 0.01 \
    --max_patience ${max_patience} \
    --gradient_accumulation_steps ${GA} \
    --eval \
    --save \
