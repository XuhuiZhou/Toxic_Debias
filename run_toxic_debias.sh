export GLUE_DIR=toxic_language
export TASK_NAME=Toxic

if [ "$1" != "" ]; then
    export BIAS_M=$1
    export RAN=$2
    export MODEL_DIR=$3
else
    export BIAS_M=ND_founta_trn_dial_pAPI_dpfbias.pkl    
    export RAN=2
    export MODEL_DIR=toxic_model_dbiasHard_s2_lr1e
fi

python run_toxic.py \
  --model_type roberta-debias \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --ensemble_bias \
  --evaluate_during_training \
  --save_steps 1000 \
  --seed $RAN \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --data_dir $GLUE_DIR/advToxicityFilt_v0.2/datamap_hard_50_ds \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --bias_model $BIAS_M \
  --output_dir /gscratch/cse/xuhuizh/$MODEL_DIR
