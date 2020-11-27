#!/bin/bash

export GLUE_DIR=toxic_language
export TASK_NAME=Toxic

if [ "$1" != "" ]; then
    export DATA=$1
    export RAN=$2
    export MODEL_DIR=$3
else
    export DATA=advToxicityFilt_v0.2/random_easy 
    export RAN=12
    export MODEL_DIR=n_toxic_model_randomeasy_s12_lr1e
fi

python run_toxic.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 1000 \
  --seed $RAN \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --data_dir $GLUE_DIR/$DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir /gscratch/cse/xuhuizh/$MODEL_DIR
