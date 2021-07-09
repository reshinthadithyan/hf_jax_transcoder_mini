python3 train_jax_crosslm.py \
    --output_dir="./tmp/crosslm" \
    --model_type="roberta" \
    --config_name="./models/crosslm" \
    --tokenizer_name="./models/crosslm" \
    --dataset_name="crosslm" \
    --dataset_config_name="unshuffled_deduplicated_als" \
    --max_seq_length="128" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="5"\
    --tpu_num_cores="8"\