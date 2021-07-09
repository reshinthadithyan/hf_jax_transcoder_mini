python3 train_jax_cross_bart.py \
    --output_dir="~/models/transcoder-js-cs/transcoder-js-cs" \
    --model_type="bart" \
    --config_name="./models/crossbart" \
    --tokenizer_name="./models/crossbart" \
    --dataset_name="crosslm" \
    --dataset_config_name="unshuffled_deduplicated_als" \
    --max_seq_length="256" \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="4" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    #--push_to_hub \
    #--push_to_hub_model_id "reshinthadithyan\transcoder-js-cs"
