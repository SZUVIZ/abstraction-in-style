#!/bin/bash

ROOT_DIR="dataset"
MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
MASK_DIR="$ROOT_DIR/mask_1024.png"

export MODEL_NAME="$MODEL_NAME"
export MASK_DIR="$MASK_DIR"
export WANDB_API_KEY="Your WANDB_API_KEY"
export WANDB_MODE="offline"

# Style folders to train.
STYLE_FOLDERS=(
    "Fluffy_Brush"
)

# Always train both stages.
TRAIN_STAGES=("A-VAT" "S-VAT")

# Stage-to-directory mapping.
declare -A TRAIN_DATA_DIRS
declare -A CHECKPOINT_DIRS

TRAIN_DATA_DIRS["A-VAT"]="A-VAT_train_Data"
TRAIN_DATA_DIRS["S-VAT"]="S-VAT_train_Data"

CHECKPOINT_DIRS["A-VAT"]="A-VAT_checkpoint"
CHECKPOINT_DIRS["S-VAT"]="S-VAT_checkpoint"

# Train each style.
for style_name in "${STYLE_FOLDERS[@]}"; do
    style_dir="$ROOT_DIR/$style_name"

    if [ ! -d "$style_dir" ]; then
        echo "Warning: $style_dir does not exist, skipping"
        continue
    fi

    for phase in "${TRAIN_STAGES[@]}"; do

        INSTANCE_DIR="$style_dir/${TRAIN_DATA_DIRS[$phase]}"
        OUTPUT_DIR="$style_dir/${CHECKPOINT_DIRS[$phase]}"

        if [ ! -d "$INSTANCE_DIR" ]; then
            echo "Warning: $INSTANCE_DIR does not exist, skipping $style_name ($phase)"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        echo "========================================"
        echo "Start training"
        echo "Style : $style_name"
        echo "Phase : $phase"
        echo "Instance dir: $INSTANCE_DIR"
        echo "Output dir  : $OUTPUT_DIR"
        echo "========================================"

        export INSTANCE_DIR
        export OUTPUT_DIR

        CUDA_VISIBLE_DEVICES=0 accelerate launch \
            --num_processes 1 \
            --main_process_port 29508 \
            ./examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora_flux.py \
            --pretrained_model_name_or_path="$MODEL_NAME" \
            --instance_data_dir="$INSTANCE_DIR" \
            --mask_data_dir="$MASK_DIR" \
            --output_dir="$OUTPUT_DIR" \
            --mixed_precision="bf16" \
            --resolution="1024" \
            --rank 16 \
            --train_batch_size=1 \
            --guidance_scale=1 \
            --gradient_accumulation_steps=4 \
            --optimizer="prodigy" \
            --learning_rate=1. \
            --report_to="wandb" \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --checkpointing_steps=500 \
            --max_train_steps=1000 \
            --num_validation_images=1 \
            --seed="42"

        echo "Completed training: $style_name ($phase)"
        echo "========================================"
        echo ""
    done
done

echo "All style and phase training jobs completed."
