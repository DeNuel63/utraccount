set -eu

PORT=${PORT:-29501}
NUM_PROCESSES=${NUM_PROCESSES:-1}
GPU_IDS=${GPU_IDS:-0}
MIXED_PRECISION=${MIXED_PRECISION:-fp16}
DATASET_CONFIG=${DATASET_CONFIG:-configs/data/ruod_256x256.py}

: "${PRETRAINED_MODEL:?Set PRETRAINED_MODEL to a local Stable Diffusion checkpoint or model identifier}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a dedicated training output directory}"
: "${RUOD_SPLIT_ROOT:?Set RUOD_SPLIT_ROOT to the reviewed split directory, not raw RUOD}"

accelerate launch --main_process_port "$PORT" --mixed_precision "$MIXED_PRECISION" \
    --gpu_ids "$GPU_IDS" --num_processes "$NUM_PROCESSES" \
    train_UWLGM.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder \
    --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
    --seed 0 --train_batch_size "${TRAIN_BATCH_SIZE:-1}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" --gradient_checkpointing \
    --mixed_precision "$MIXED_PRECISION" --num_train_epochs 60 --learning_rate 1.5e-4 --max_grad_norm 1 \
    --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine --lr_warmup_steps 3000 \
    --dataset_config_name "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --uncond_prob 0.1 \
    "$@"
