#!/usr/bin/env python
# coding=utf-8
# Modified from original diffusers dreambooth training script for multi-context support
# UPDATED: Full parameter fine-tuning instead of LoRA

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from accelerate.utils import DistributedType

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")
logger = get_logger(__name__)


# Preferred resolutions for bucketing
PREFERRED_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672)
]


class MultiContextDataset(Dataset):
    def __init__(self, root_dir, max_context_images=6):
        self.root_dir = root_dir
        self.max_context_images = max_context_images
        
        self.samples = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('sample_')
        ])
        
        self.PREFERRED_RESOLUTIONS = PREFERRED_RESOLUTIONS
        
    def find_best_resolution(self, img):
        w, h = img.size
        aspect_ratio = w / h
        
        best_match = None
        min_diff = float('inf')
        
        for res_w, res_h in self.PREFERRED_RESOLUTIONS:
            res_ar = res_w / res_h
            diff = abs(aspect_ratio - res_ar)
            if diff < min_diff:
                min_diff = diff
                best_match = (res_w, res_h)
        
        return best_match
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        
        # Load prompt
        with open(os.path.join(sample_dir, 'prompt.txt'), 'r') as f:
            prompt = f.read().strip()
        
        # Load and process target image
        target_path = os.path.join(sample_dir, 'out.jpg')
        if not os.path.exists(target_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(sample_dir, f'out{ext}')
                if os.path.exists(alt_path):
                    target_path = alt_path
                    break
        
        target_img = Image.open(target_path).convert('RGB')
        target_w, target_h = self.find_best_resolution(target_img)
        target_img = target_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        target_tensor = transform(target_img)
        
        # Load context images
        context_tensors = []
        context_dir = os.path.join(sample_dir, 'in')
        if os.path.exists(context_dir):
            context_files = sorted([
                f for f in os.listdir(context_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ])
            for img_file in context_files[:self.max_context_images]:
                img = Image.open(os.path.join(context_dir, img_file)).convert('RGB')
                ctx_w, ctx_h = self.find_best_resolution(img)
                img = img.resize((ctx_w, ctx_h), Image.Resampling.LANCZOS)
                context_tensors.append(transform(img))
        
        return {
            "txt": prompt,
            "img": target_tensor,
            "context_images": context_tensors,
        }

def process_context_images_spatial(context_tensors, vae, vae_config_shift_factor, 
                                  vae_config_scaling_factor, args, accelerator, weight_dtype):
    """Process context images using spatial offset method - horizontal stacking only"""
    context_latents_list = []
    context_ids_list = []
    
    # Reverse the order so most recent (highest numbered) images come first
    # This assumes your dataset provides them in ascending order (1.jpg, 2.jpg, etc.)
    context_tensors = list(reversed(context_tensors[:args.max_context_images]))
    
    # Track accumulated width for horizontal stacking
    w_accumulated = 0
    
    for idx, ctx_tensor in enumerate(context_tensors):
        ctx_tensor = ctx_tensor.unsqueeze(0).to(device=accelerator.device, dtype=weight_dtype)
        
        # Encode
        if args.vae_encode_mode == "sample":
            ctx_latent = vae.encode(ctx_tensor).latent_dist.sample()
        else:
            ctx_latent = vae.encode(ctx_tensor).latent_dist.mode()
        
        ctx_latent = (ctx_latent - vae_config_shift_factor) * vae_config_scaling_factor
        ctx_latent = ctx_latent.to(dtype=weight_dtype)
        
        h, w = ctx_latent.shape[2:]
        packed = FluxKontextPipeline._pack_latents(
            ctx_latent, 1, ctx_latent.shape[1], h, w
        )
        context_latents_list.append(packed)
        
        # Prepare IDs with horizontal offsets only
        ids = FluxKontextPipeline._prepare_latent_image_ids(
            1, h // 2, w // 2, accelerator.device, weight_dtype
        )
        
        # All context at tau=1, stack horizontally from left to right
        ids[..., 0] = 1.0  # tau = 1 for all context
        ids[..., 1] += 0    # No vertical offset (y stays at 0)
        ids[..., 2] += w_accumulated  # Horizontal offset based on accumulated width
        
        w_accumulated += w // 2  # Add half-width for next image positioning
        
        context_ids_list.append(ids)
    
    return context_latents_list, context_ids_list

def letterbox_image(img: Image.Image, target_size: Tuple[int, int], color=(245, 245, 245)) -> Image.Image:
    """Resize image preserving aspect ratio and pad to target size"""
    w, h = img.size
    target_w, target_h = target_size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize preserving aspect ratio
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new image with padding
    img_padded = Image.new('RGB', target_size, color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))
    
    return img_padded


def create_validation_collage(
    context_images: List[Image.Image],
    target_img: Image.Image,
    base_output: Image.Image,
    finetuned_output: Image.Image,
    sample_name: str,
    step: int
) -> Image.Image:
    """Create a simple horizontal collage"""
    
    # Fixed sizes
    main_h = 512
    ctx_h = 256
    padding = 10
    
    # Resize preserving aspect ratio
    def resize_to_height(img, target_h):
        w, h = img.size
        new_w = int(w * target_h / h)
        return img.resize((new_w, target_h), Image.Resampling.LANCZOS)
    
    # Resize all images to fixed heights
    target_resized = resize_to_height(target_img, main_h)
    base_resized = resize_to_height(base_output, main_h)
    finetuned_resized = resize_to_height(finetuned_output, main_h)
    context_resized = [resize_to_height(ctx, ctx_h) for ctx in context_images[:4]]  # Limit to 4 for space
    
    # Calculate total width
    ctx_total_w = sum(img.size[0] for img in context_resized) + len(context_resized) * padding
    main_total_w = target_resized.size[0] + base_resized.size[0] + finetuned_resized.size[0] + 3 * padding
    
    # Create canvas
    total_w = max(ctx_total_w, main_total_w) + 2 * padding
    total_h = 80 + ctx_h + padding + main_h + 60
    
    grid = Image.new('RGB', (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    
    # Fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        bold_font = font
    
    # Title
    draw.text((padding, 10), f"{sample_name} - Step {step}", fill='black', font=bold_font)
    
    # Context images row
    draw.text((padding, 40), "Context:", fill='gray', font=font)
    x_offset = padding
    for ctx in context_resized:
        grid.paste(ctx, (x_offset, 60))
        x_offset += ctx.size[0] + padding
    
    # Main comparison row
    y_main = 60 + ctx_h + 30
    draw.text((padding, y_main - 20), "Target → Base → Finetuned:", fill='gray', font=font)
    
    x_offset = padding
    grid.paste(target_resized, (x_offset, y_main))
    draw.text((x_offset + target_resized.size[0]//2 - 20, y_main + main_h + 5), "Target", fill='black', font=font)
    
    x_offset += target_resized.size[0] + padding
    grid.paste(base_resized, (x_offset, y_main))
    draw.text((x_offset + base_resized.size[0]//2 - 20, y_main + main_h + 5), "Base", fill='blue', font=font)
    
    x_offset += base_resized.size[0] + padding
    grid.paste(finetuned_resized, (x_offset, y_main))
    draw.text((x_offset + finetuned_resized.size[0]//2 - 30, y_main + main_h + 5), "Finetuned", fill='green', font=font)
    
    return grid


@torch.no_grad()
def precompute_base_outputs(validation_samples, base_transformer, text_encoder_one, text_encoder_two,
                           vae, tokenizer_one, tokenizer_two, noise_scheduler_copy, args, weight_dtype, accelerator):
    """Pre-compute base model outputs for all validation samples"""
    
    logger.info("Pre-computing base model outputs for validation samples...")
    base_outputs = []
    
    # Put models in eval mode
    base_transformer.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()
    
    # Get VAE config values
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
    
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux.pipeline_flux_img2img import calculate_shift
    
    for sample_idx, sample in enumerate(validation_samples):
        logger.info(f"Pre-computing base output for sample {sample_idx + 1}/{len(validation_samples)}")
        
        prompt = sample["txt"]
        target_tensor = sample["img"]
        context_tensors = sample["context_images"]
        
        # Get dimensions
        target_h, target_w = target_tensor.shape[1], target_tensor.shape[2]
        
        # Text encoding
        text_inputs_clip = tokenizer_one(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        clip_output = text_encoder_one(
            text_inputs_clip.input_ids.to(accelerator.device),
            output_hidden_states=False
        )
        pooled_prompt_embeds = clip_output.pooler_output.to(dtype=weight_dtype)
        
        text_inputs_t5 = tokenizer_two(
            prompt,
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
            return_tensors="pt"
        )
        prompt_embeds = text_encoder_two(
            text_inputs_t5.input_ids.to(accelerator.device),
            output_hidden_states=False
        )[0].to(dtype=weight_dtype)
        
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
        
        # Process context images with spatial method
        if len(context_tensors) > 0:
            # Note: args.vae_encode_mode is always "mode" for precompute
            temp_args = args
            temp_vae_mode = args.vae_encode_mode
            args.vae_encode_mode = "mode"  # Force mode for precompute
            context_latents_list, context_ids_list = process_context_images_spatial(
                context_tensors, vae, vae_config_shift_factor, 
                vae_config_scaling_factor, args, accelerator, weight_dtype
            )
            args.vae_encode_mode = temp_vae_mode  # Restore
        else:
            context_latents_list = []
            context_ids_list = []
        
        # Prepare noise with fixed seed for consistency
        generator = torch.Generator(device="cpu").manual_seed(42 + sample_idx)
        target_shape = (1, vae.config.latent_channels, target_h // vae_scale_factor, target_w // vae_scale_factor)
        latents = torch.randn(target_shape, generator=generator, dtype=torch.float32)
        latents = latents.to(device=accelerator.device, dtype=weight_dtype)
        packed_latents = FluxKontextPipeline._pack_latents(
            latents, 1, latents.shape[1], latents.shape[2], latents.shape[3]
        )
        
        target_ids = FluxKontextPipeline._prepare_latent_image_ids(
            1, latents.shape[2] // 2, latents.shape[3] // 2,
            accelerator.device, weight_dtype
        )
        target_ids[..., 0] = 0
        
        combined_latents = torch.cat([packed_latents] + context_latents_list, dim=1)
        combined_ids = torch.cat([target_ids] + context_ids_list, dim=0)
        
        # Guidance
        if getattr(base_transformer.config, "guidance_embeds", False):
            guidance = torch.tensor([args.guidance_scale], device=accelerator.device, dtype=weight_dtype)
        else:
            guidance = None
        
        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        
        image_seq_len = packed_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        
        sigmas = np.linspace(1.0, 1 / args.validation_inference_steps, args.validation_inference_steps)
        scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=accelerator.device)
        timesteps = scheduler.timesteps
        
        # Denoising loop
        for t in timesteps:
            timestep = t.expand(packed_latents.shape[0]).to(dtype=weight_dtype)
            
            noise_pred = base_transformer(
                hidden_states=combined_latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0]
            
            noise_pred = noise_pred[:, :packed_latents.size(1)]
            packed_latents = scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
            combined_latents = torch.cat([packed_latents] + context_latents_list, dim=1)
        
        # Decode
        final_latents = FluxKontextPipeline._unpack_latents(
            packed_latents, height=target_h, width=target_w, vae_scale_factor=vae_scale_factor,
        )
        final_latents = (final_latents / vae_config_scaling_factor) + vae_config_shift_factor
        final_latents = final_latents.to(dtype=weight_dtype)
        
        image = vae.decode(final_latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        base_outputs.append(Image.fromarray(image[0]))
    
    logger.info(f"Pre-computed {len(base_outputs)} base model outputs")
    return base_outputs


@torch.no_grad()
def run_validation(step, accelerator, transformer, text_encoder_one, text_encoder_two,
                  vae, tokenizer_one, tokenizer_two, noise_scheduler_copy, 
                  validation_samples, base_outputs, args, weight_dtype):
    """FSDP-compatible validation - ALL processes participate but only main saves"""
    
    # ALL processes must participate in validation with FSDP
    # Remove the early return that was causing the hang
    
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux.pipeline_flux_img2img import calculate_shift
    
    logger.info(f"Running validation at step {step} on process {accelerator.process_index}")
    
    # Create step directory (only main process)
    if accelerator.is_main_process:
        step_dir = os.path.join(args.output_dir, "validation", f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
    
    # VAE config
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    
    # Process each validation sample
    for sample_idx in range(min(args.num_validation_samples, len(validation_samples))):
        sample = validation_samples[sample_idx]
        base_output = base_outputs[sample_idx] if sample_idx < len(base_outputs) else None
        
        try:
            logger.info(f"Processing validation sample {sample_idx + 1}/{args.num_validation_samples}")
            
            prompt = sample["txt"]
            target_tensor = sample["img"]
            context_tensors = sample["context_images"]
            
            target_h, target_w = target_tensor.shape[1], target_tensor.shape[2]
            
            # Text encoding - all processes participate
            with accelerator.autocast():
                text_inputs_clip = tokenizer_one(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                clip_output = text_encoder_one(text_inputs_clip.input_ids.to(accelerator.device), output_hidden_states=False)
                pooled_prompt_embeds = clip_output.pooler_output.to(dtype=weight_dtype)
                
                text_inputs_t5 = tokenizer_two(prompt, padding="max_length", max_length=args.max_sequence_length, truncation=True, return_tensors="pt")
                prompt_embeds = text_encoder_two(text_inputs_t5.input_ids.to(accelerator.device), output_hidden_states=False)[0].to(dtype=weight_dtype)
            
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
            
            # Process context images with spatial method
            if len(context_tensors) > 0:
                # Note: validation always uses mode
                temp_args = args
                temp_vae_mode = args.vae_encode_mode
                args.vae_encode_mode = "mode"  # Force mode for validation
                context_latents_list, context_ids_list = process_context_images_spatial(
                    context_tensors, vae, vae_config_shift_factor, 
                    vae_config_scaling_factor, args, accelerator, weight_dtype
                )
                args.vae_encode_mode = temp_vae_mode  # Restore
            else:
                context_latents_list = []
                context_ids_list = []
            
            # Prepare noise with same seed as base for fair comparison
            generator = torch.Generator(device="cpu").manual_seed(42 + sample_idx)
            target_shape = (1, vae.config.latent_channels, target_h // vae_scale_factor, target_w // vae_scale_factor)
            latents = torch.randn(target_shape, generator=generator, dtype=torch.float32)
            latents = latents.to(device=accelerator.device, dtype=weight_dtype)
            packed_latents = FluxKontextPipeline._pack_latents(latents, 1, latents.shape[1], latents.shape[2], latents.shape[3])
            
            target_ids = FluxKontextPipeline._prepare_latent_image_ids(1, latents.shape[2] // 2, latents.shape[3] // 2, accelerator.device, weight_dtype)
            target_ids[..., 0] = 0
            
            combined_latents = torch.cat([packed_latents] + context_latents_list, dim=1)
            combined_ids = torch.cat([target_ids] + context_ids_list, dim=0)
            
            # Guidance
            guidance = torch.tensor([args.guidance_scale], device=accelerator.device, dtype=weight_dtype) if args.guidance_scale > 0 else None
            
            # Scheduler with full validation_inference_steps
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            image_seq_len = packed_latents.shape[1]
            mu = calculate_shift(image_seq_len, scheduler.config.base_image_seq_len, scheduler.config.max_image_seq_len, 
                               scheduler.config.base_shift, scheduler.config.max_shift)
            
            sigmas = np.linspace(1.0, 1 / args.validation_inference_steps, args.validation_inference_steps)
            scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=accelerator.device)
            
            # Denoising loop - ALL processes must participate
            with accelerator.autocast():
                for i, t in enumerate(scheduler.timesteps):
                    if i % 10 == 0 and accelerator.is_main_process:
                        logger.info(f"Sample {sample_idx + 1}: Denoising step {i}/{len(scheduler.timesteps)}")
                    
                    timestep = t.expand(packed_latents.shape[0]).to(dtype=weight_dtype)
                    
                    # ALL processes run the transformer
                    noise_pred = transformer(
                        hidden_states=combined_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=combined_ids,
                        return_dict=False,
                    )[0]
                    
                    noise_pred = noise_pred[:, :packed_latents.size(1)]
                    packed_latents = scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
                    combined_latents = torch.cat([packed_latents] + context_latents_list, dim=1)
            
            # Only main process decodes and saves images
            if accelerator.is_main_process:
                # Decode
                final_latents = FluxKontextPipeline._unpack_latents(packed_latents, height=target_h, width=target_w, vae_scale_factor=vae_scale_factor)
                final_latents = (final_latents / vae_config_scaling_factor) + vae_config_shift_factor
                final_latents = final_latents.to(dtype=vae.dtype)
                
                image = vae.decode(final_latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.float().cpu().permute(0, 2, 3, 1).numpy()
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
                finetuned_output = Image.fromarray(image[0])
                
                # Convert target tensor to PIL
                target_array = ((target_tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                target_pil = Image.fromarray(target_array)
                
                # Convert context tensors to PIL
                context_pils = []
                for ctx_tensor in context_tensors[:6]:  # Max 6 for display
                    ctx_array = ((ctx_tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    context_pils.append(Image.fromarray(ctx_array))
                
                # Create collage using pre-computed base output
                if base_output is not None:
                    collage = create_validation_collage(
                        context_images=context_pils,
                        target_img=target_pil,
                        base_output=base_output,
                        finetuned_output=finetuned_output,
                        sample_name=f"sample_{sample_idx:03d}",
                        step=step
                    )
                else:
                    # Fallback if no base output - simple side by side
                    collage = Image.new('RGB', (target_pil.width + finetuned_output.width + 10, max(target_pil.height, finetuned_output.height)), (255, 255, 255))
                    collage.paste(target_pil, (0, 0))
                    collage.paste(finetuned_output, (target_pil.width + 10, 0))
                
                # Save collage
                collage_path = os.path.join(step_dir, f"sample_{sample_idx:03d}.jpg")
                collage.save(collage_path)
                logger.info(f"Saved validation collage to {collage_path}")
                
                # Log to wandb
                if is_wandb_available():
                    wandb.log({
                        f"validation/sample_{sample_idx}": wandb.Image(collage, caption=f"Step {step} - Sample {sample_idx}"),
                        "global_step": step,
                    })
                    
        except Exception as e:
            logger.warning(f"Validation failed for sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Synchronize all processes before returning
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info(f"Validation complete. Results saved to {step_dir}")


def get_validation_samples(dataset, num_samples=3, seed=42):
    """Get fixed validation samples for consistent evaluation"""
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:min(num_samples, len(dataset))].tolist()
    return [dataset[i] for i in indices]


def save_model_card(repo_id, images=None, base_model=None, train_text_encoder=False,
                   instance_prompt=None, validation_prompt=None, repo_folder=None):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append({
                "text": validation_prompt if validation_prompt else " ",
                "output": {"url": f"image_{i}.png"}
            })

    model_description = f"""
# Flux Kontext Multi-Context Full Fine-tune - {repo_id}

## Model description
Full parameter fine-tuned weights for {base_model} trained with multiple reference images.

Was the text encoder fine-tuned? {train_text_encoder}.

## Trigger words
You should use `{instance_prompt}` to trigger the image generation.

## Download model
[Download]({repo_id}/tree/main) the full model weights in the Files & versions tab.
"""
    
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    
    tags = ["text-to-image", "diffusers-training", "diffusers", "flux", "flux-kontext", "full-finetune"]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_length=False, return_overflowing_tokens=False,
        return_tensors="pt"
    )
    return text_inputs.input_ids


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length=512,
                          prompt=None, num_images_per_prompt=1, device=None, text_input_ids=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_length=False, return_overflowing_tokens=False,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None,
                            text_input_ids=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True,
            return_overflowing_tokens=False, return_length=False, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length,
                 device=None, num_images_per_prompt=1, text_input_ids_list=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0], tokenizer=tokenizers[0],
        prompt=prompt, device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1], tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length, prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Multi-context Flux Kontext full fine-tuning script.")
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
                       help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None,
                       help="Variant of the model files of the pretrained model identifier")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="data/train",
                       help="Directory containing training data")
    parser.add_argument("--max_context_images", type=int, default=6,
                       help="Maximum number of context images to use")
    # parser.add_argument("--time_spacing", type=float, default=1.0,
    #                    help="Spacing between tau values for context images")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="flux-kontext-full-finetune",
                       help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=1024,
                       help="The resolution for input images")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                       help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                       help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Whether training should be resumed from a previous checkpoint.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Whether or not to use gradient checkpointing to save memory")
    
    # Text encoder training
    parser.add_argument("--train_text_encoder", action="store_true",
                       help="Whether to train the text encoder")
    
    # Optimizer arguments
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                       help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                       help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0,
                       help="Power factor of the polynomial scheduler.")
    
    parser.add_argument("--optimizer", type=str, default="AdamW",
                       help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--use_8bit_adam", action="store_true",
                       help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04,
                       help="Weight decay to use for unet params")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                       help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                       help="Max gradient norm.")
    
    # Other arguments
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                       help="Guidance scale for generation")
    parser.add_argument("--vae_encode_mode", type=str, default="mode", choices=["sample", "mode"],
                       help="VAE encoding mode.")
    parser.add_argument("--weighting_scheme", type=str, default="none",
                       choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
                       help='Weighting scheme for loss')
    parser.add_argument("--logit_mean", type=float, default=0.0,
                       help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0,
                       help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29,
                       help="Scale of mode weighting scheme.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of subprocesses to use for data loading.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                       help="Whether to use mixed precision.")
    parser.add_argument("--allow_tf32", action="store_true",
                       help="Whether or not to allow TF32 on Ampere GPUs.")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       help='The integration to report the results and logs to.')
    parser.add_argument("--validation_prompt", type=str, default=None,
                       help="A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument("--num_validation_images", type=int, default=4,
                       help="Number of images that should be generated during validation")
    parser.add_argument("--validation_epochs", type=int, default=50,
                       help="Run validation every X epochs.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                       help="TensorBoard log directory.")
    parser.add_argument("--hub_token", type=str, default=None,
                       help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                       help="Maximum sequence length to use with the T5 text encoder")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="For distributed training: local_rank")
    
    # Validation arguments
    parser.add_argument("--validation_steps", type=int, default=500,
                       help="Run validation every X steps")
    parser.add_argument("--num_validation_samples", type=int, default=3,
                       help="Number of samples to validate")
    parser.add_argument("--validation_inference_steps", type=int, default=20,
                       help="Inference steps for validation (less = faster)")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Create dataset
    train_dataset = MultiContextDataset(
        root_dir=args.data_dir,
        max_context_images=args.max_context_images
    )
    
    # Get validation samples
    validation_samples = get_validation_samples(train_dataset, args.num_validation_samples)
    logger.info(f"Selected {len(validation_samples)} validation samples")

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    # Import text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant
    )
    
    # Load transformer with proper dtype from start to avoid conversion issues
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant,
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    )

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Pre-compute base model outputs before enabling gradients
    base_outputs = precompute_base_outputs(
        validation_samples, transformer, text_encoder_one, text_encoder_two,
        vae, tokenizer_one, tokenizer_two, noise_scheduler_copy, args, weight_dtype, accelerator
    )
    
    # Save base outputs for reproducibility
    if accelerator.is_main_process:
        base_outputs_dir = os.path.join(args.output_dir, "base_outputs")
        os.makedirs(base_outputs_dir, exist_ok=True)
        for idx, img in enumerate(base_outputs):
            img.save(os.path.join(base_outputs_dir, f"sample_{idx:03d}.png"))
        logger.info(f"Saved base outputs to {base_outputs_dir}")

    # Enable training for full fine-tuning
    transformer.requires_grad_(True)
    vae.requires_grad_(False)  # Keep VAE frozen
    text_encoder_two.requires_grad_(False)  # Keep T5 frozen
    
    if args.train_text_encoder:
        text_encoder_one.requires_grad_(True)
    else:
        text_encoder_one.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # Set up optimizer for all trainable parameters
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * 
            args.train_batch_size * accelerator.num_processes
        )
    
    # Collect all parameters to optimize
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        params_to_optimize.extend(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # Use more stable optimizer settings for FSDP
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=1e-6,  # Increased from 1e-8 for stability
        foreach=False,  # Disable foreach optimization for FSDP compatibility
    )

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )

    # Compute text embeddings if not training text encoder
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # Get VAE config values
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

    # Set up scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with accelerator
    if args.train_text_encoder:
        transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # Set up trackers
    if accelerator.is_main_process:
        tracker_name = "flux-kontext-multi-full-finetune"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Training loop with debugging
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
        
        # Track initial state
        if global_step == 0 and accelerator.is_main_process:
            initial_param = next(transformer.parameters()).clone().detach()
            logger.info(f"🔍 Initial param sample - mean: {initial_param.mean():.6f}, std: {initial_param.std():.6f}")
    
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
                
            with accelerator.accumulate(models_to_accumulate):
                # Since batch_size=1, extract the single sample
                prompts = batch["txt"]
                target_images = batch["img"]
                context_images_list = batch["context_images"][0] if batch["context_images"] else []
    
                # Encode prompts
                if not args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts, max_sequence_length=args.max_sequence_length)
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[tokens_one, tokens_two],
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        prompt=prompts,
                    )
    
                # Encode target images
                pixel_values = target_images.to(dtype=vae.dtype)
                if args.vae_encode_mode == "sample":
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                else:
                    model_input = vae.encode(pixel_values).latent_dist.mode()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
    
                # Process context images with spatial method
                if len(context_images_list) > 0:
                    context_latents_list, context_ids_list = process_context_images_spatial(
                        context_images_list, vae, vae_config_shift_factor, 
                        vae_config_scaling_factor, args, accelerator, weight_dtype
                    )
                else:
                    context_latents_list = []
                    context_ids_list = []
    
                # Sample noise and timesteps
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
    
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    
                # Prepare latent IDs for target
                latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                # Combine all IDs
                id_parts = [latent_ids]
                if context_ids_list:
                    id_parts.extend(context_ids_list)
                combined_latent_ids = torch.cat(id_parts, dim=0)
                
                # Add noise
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                
                # Pack noisy input
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                
                # Prepare guidance
                unwrapped = accelerator.unwrap_model(transformer)
                if getattr(unwrapped.config, "guidance_embeds", False):
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None
                
                # Combine all latents
                latent_parts = [packed_noisy_model_input]
                if context_latents_list:
                    latent_parts.extend(context_latents_list)
                latent_model_input = torch.cat(latent_parts, dim=1)
                
                # Forward pass
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_latent_ids,
                    return_dict=False,
                )[0]
                
                # Extract only target prediction
                model_pred = model_pred[:, :packed_noisy_model_input.size(1)]
                
                # Unpack
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Compute loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"❌ Invalid loss at step {global_step}: {loss.item()}")
                    optimizer.zero_grad()
                    continue
                
                # Backward pass
                accelerator.backward(loss)
                
                # Debug: Check gradients after backward (every 20 steps)
                if accelerator.sync_gradients and global_step % 20 == 0 and accelerator.is_main_process:
                    grad_norm = 0.0
                    param_norm = 0.0
                    num_params_with_grad = 0
                    
                    for p in transformer.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                            param_norm += p.data.norm(2).item() ** 2
                            num_params_with_grad += 1
                    
                    grad_norm = grad_norm ** 0.5
                    param_norm = param_norm ** 0.5
                    
                    if grad_norm == 0:
                        logger.warning(f"⚠️ Step {global_step}: ZERO gradients detected!")
                    else:
                        logger.info(f"📊 Step {global_step}: grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f}, num_grads={num_params_with_grad}")
                
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
    
                # Store params before optimizer step (for comparison)
                if global_step % 20 == 0 and accelerator.is_main_process:
                    sample_param_before = next(transformer.parameters()).clone().detach()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Check if params actually changed (every 20 steps)
                if global_step % 20 == 0 and accelerator.is_main_process:
                    sample_param_after = next(transformer.parameters()).clone().detach()
                    param_change = (sample_param_after - sample_param_before).abs().max().item()
                    
                    if param_change < 1e-8:
                        logger.error(f"❌ Step {global_step}: Parameters DID NOT change! Change={param_change:.2e}")
                    else:
                        logger.info(f"✅ Step {global_step}: Parameters changed by {param_change:.6f}")
    
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log key metrics every 10 steps
                if global_step % 10 == 0 and accelerator.is_main_process:
                    actual_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"📈 Step {global_step}: loss={loss.item():.4f}, lr={actual_lr:.2e}")
                
                # Run validation periodically
                if global_step % args.validation_steps == 0:
                    logger.info(f"🎨 Starting validation at step {global_step}")
                    run_validation(
                        global_step, accelerator, transformer,
                        text_encoder_one, text_encoder_two, vae,
                        tokenizer_one, tokenizer_two, noise_scheduler_copy,
                        validation_samples, base_outputs, args, weight_dtype
                    )
                    
                    # After validation, check if model is still in training mode
                    if accelerator.is_main_process:
                        logger.info(f"Post-validation check - transformer.training={transformer.training}")
    
                # Checkpoint saving (simplified - only main process checks)
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"💾 Saved checkpoint to {save_path}")
    
            logs = {"loss": loss.detach().item()}
            if lr_scheduler is not None:
                logs["lr"] = lr_scheduler.get_last_lr()[0]
            elif accelerator.distributed_type == DistributedType.DEEPSPEED:
                logs["lr"] = accelerator.unwrap_model(transformer).optimizer.param_groups[0]['lr']
            else:
                logs["lr"] = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
    
            if global_step >= args.max_train_steps:
                break
                
        if global_step >= args.max_train_steps:
            break
    
    # Final parameter check
    if accelerator.is_main_process and global_step > 0:
        final_param = next(transformer.parameters()).clone().detach()
        total_change = (final_param - initial_param).abs().max().item()
        logger.info(f"🏁 Training complete. Total param change from start: {total_change:.6f}")
        if total_change < 1e-6:
            logger.error("❌ WARNING: Model parameters barely changed during training!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
