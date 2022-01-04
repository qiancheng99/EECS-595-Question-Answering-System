export BASE_MODEL=albert-base-v2
export OUTPUT_MODEL=./output/
export SQUAD_DIR=../squad
python run_squad.py \
  --version_2_with_negative \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --output_dir ./output/ \
  --do_eval \
  --do_lower_case \
  --train_file ../squad/train-v2.0.json \
  --predict_file ../squad/dev-v2.0.json \
  --per_gpu_train_batch_size 3 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 2000 \
  --threads 24 \
  --warmup_steps 814 \
  --gradient_accumulation_steps 4 \
  --do_train

