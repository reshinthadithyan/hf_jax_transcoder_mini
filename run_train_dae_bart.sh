python3 train_dae_bart.py \
    --output_dir="~/models/transcoder-dae-js-cs" \
    --model_type="bart" \
    --model_name_or_path="reshinthadith/transcoder-crossbartlm-js-cs" \
    --tokenizer_name="reshinthadith/transcoder-crossbartlm-js-cs" \
    --dataset_name="crosslm" \
    --dataset_config_name="unshuffled_deduplicated_als" \
    --max_source_length="256" \
    --max_target_length="256"\
    --do_train\
    --do_eval\
    --do_predict\
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="4" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="35" \
    --push_to_hub \
    --push_to_hub_model_id "reshinthadith/transcoder-dae-js-cs"
