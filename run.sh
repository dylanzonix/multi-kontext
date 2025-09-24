#!/bin/bash
# export HF_HOME=$(pwd)/cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1

accelerate launch train.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev" \
  --output_dir="kontext-full-finetune" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1.0 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --report_to="wandb" \
  --data_dir="/workspace/data/train" \
  --learning_rate=1e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --lr_num_cycles=3 \
  --adam_beta2=0.95 \
  --adam_weight_decay=0.01 \
  --max_grad_norm=1.0 \
  --num_train_epochs=999999 \
  --max_train_steps=5000 \
  --checkpointing_steps=999999 \
  --dataloader_num_workers=0 \
  --max_sequence_length=512 \
  --weighting_scheme="logit_normal" \
  --logit_mean=1.1 \
  --logit_std=1.0 \
  --seed=42 \
  --validation_steps=100 \
  --num_validation_samples=4 \
  --validation_inference_steps=30
