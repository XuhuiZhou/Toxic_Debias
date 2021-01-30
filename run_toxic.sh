#!/bin/bash

export TOXIC_DIR=/toxic/language/data
export TASK_NAME=Toxic

export DATA=$1
export RAN=$2
export MODEL_DIR=$3

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
  --data_dir $TOXIC_DIR/$DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir $MODEL_DIR
