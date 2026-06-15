#!/bin/bash

ROOT_DIR="dataset"
MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
MASK_DIR="$ROOT_DIR/mask_1024.png"

export MODEL_NAME="$MODEL_NAME"
export MASK_DIR="$MASK_DIR"
export WANDB_API_KEY="Your WANDB_API_KEY"
export WANDB_MODE="offline"

# Styles to train, in order.
STYLE_ORDER=(
    "Fluffy_Brush"
)

# Always train both phases.
TRAIN_PHASES=("A-VAT" "S-VAT")

# Phase-to-directory mapping.
declare -A INSTANCE_SUFFIX
declare -A OUTPUT_SUFFIX

INSTANCE_SUFFIX["A-VAT"]="A-VAT_train_Data"
INSTANCE_SUFFIX["S-VAT"]="S-VAT_train_Data"

OUTPUT_SUFFIX["A-VAT"]="A-VAT_checkpoint"
OUTPUT_SUFFIX["S-VAT"]="S-VAT_checkpoint"

# Train each style in order.
for style_name in "${STYLE_ORDER[@]}"; do
    style_dir="$ROOT_DIR/$style_name"

    if [ ! -d "$style_dir" ]; then
        echo "Warning: $style_dir does not exist, skipping"
        continue
    fi

    for mode in "${TRAIN_PHASES[@]}"; do

        INSTANCE_DIR="$style_dir/${INSTANCE_SUFFIX[$mode]}"
        OUTPUT_DIR="$style_dir/${OUTPUT_SUFFIX[$mode]}"

        if [ ! -d "$INSTANCE_DIR" ]; then
            echo "Warning: $INSTANCE_DIR does not exist, skipping $style_name ($mode)"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        echo "========================================"
        echo "Start training"
        echo "STYLE : $style_name"
        echo "MODE  : $mode"
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

        echo "Completed training: $style_name ($mode)"
        echo "========================================"
        echo ""
    done
done

echo "All style and phase training jobs completed."
