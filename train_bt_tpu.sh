python3 train_bt_tpu.py \
    --output_dir="~/models/transcoder-backtrans-js-cs" \
    --model_type="bart" \
    --model_name_or_path="reshinthadith/transcoder-dae-js-cs" \
    --tokenizer_name="reshinthadith/transcoder-dae-js-cs" \
    --dataset_name="crosslm" \
    --dataset_config_name="unshuffled_deduplicated_als" \
    --max_source_length="256" \
    --max_target_length="256"\
    --do_train\
    --do_eval\
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --save_steps="5"\
    --overwrite_output_dir \
    --num_train_epochs="5" \
    --push_to_hub \
    --push_to_hub_model_id "reshinthadith/transcoder-java-cs"
