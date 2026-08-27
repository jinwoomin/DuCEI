#!/bin/bash

lm_name="allenai/longformer-large-4096"
tok_name="longformer_tok"
max_seq_len=1024
opt=3
seed=1013
data_path="./datasets"

# Directory containing the Wikipedia2Vec DumpDB and processed entity information.
wikipedia_path="./wikipedia"

save_path="data/stage1/instances_${tok_name}_opt${opt}_seq_len${max_seq_len}_seed${seed}"
python preprocess.py \
                    --data_path ${data_path} \
                    --wikipedia_path ${wikipedia_path} \
                    --example_file "examples.pkl" \
                    --save_path ${save_path} \
                    --epochs 5 \
                    --max_seq_len ${max_seq_len} \
                    --opt ${opt} \
                    --seed ${seed} \
