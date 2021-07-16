python3 train_dae_bart.py \
    --output_dir="~/models/transcoder-dae-jc" \
    --model_type="bart" \
    --model_name_or_path="reshinthadith/transcoder-encoder-mlm-jc" \
    --tokenizer_name="reshinthadith/transcoder-encoder-mlm-jc" \
    --dataset_name="crosslm" \
    --dataset_config_name="unshuffled_deduplicated_als" \
    --max_source_length="512" \
    --max_target_length="512"\
    --do_train\
    --do_eval\
    --do_predict\
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="35" \
    --push_to_hub \
    --push_to_hub_model_id "reshinthadith/transcoder-dae-jc"
