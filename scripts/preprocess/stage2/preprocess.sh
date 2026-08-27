#!/bin/bash

lm_name="allenai/longformer-large-4096"
tok_name="longformer_tok"
max_seq_len=1024
opt=3
seed=1013
k=10
sampling='probability'

# Directory containing the Wikipedia2Vec DumpDB and processed entity information.
wikipedia_path="./wikipedia"

# Directory containing stage-1 prediction files used as input for second-step preprocessing.
# The files should follow the same KILT-style JSONL format expected by preprocess.py with --second_step.
stage1_prediction_path="./stage1_predictions"

save_path="data/test_instances_${tok_name}_opt${opt}_seq_len${max_seq_len}_seed${seed}_second_step_k${k}_sampling_${sampling}_add_desc_add_type"
python preprocess.py \
            --example_file "examples.pkl" \
            --data_path ${stage1_prediction_path} \
            --wikipedia_path ${wikipedia_path} \
            --save_path ${save_path} \
            --epochs 5 \
            --max_seq_len ${max_seq_len} \
            --opt ${opt} \
            --seed ${seed} \
            --second_step \
            --max_num_candidates ${k} \
            --sampling ${sampling} \
            --add_desc \
            --add_type \
