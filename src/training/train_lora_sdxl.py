"""
LoRA fine-tuning script for Stable Diffusion XL.

Reads a dataset of images + metadata.jsonl captions, trains a LoRA adapter
on the UNet (and optionally text encoder), and saves the weights.

Launched via: accelerate launch src/training/train_lora_sdxl.py --args...
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CaptionedImageDataset(Dataset):
    """Reads images + captions from a directory with metadata.jsonl."""

    def __init__(self, data_dir: str, resolution: int, tokenizer_1, tokenizer_2):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

        self.entries = []
        with open(self.data_dir / "metadata.jsonl") as f:
            for line in f:
                self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(self.data_dir / entry["file_name"]).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Normalize to [-1, 1]
        image = torch.tensor(
            list(image.getdata()), dtype=torch.float32
        ).reshape(self.resolution, self.resolution, 3) / 127.5 - 1.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        caption = entry["text"]

        tokens_1 = self.tokenizer_1(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer_1.model_max_length, return_tensors="pt"
        )
        tokens_2 = self.tokenizer_2(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer_2.model_max_length, return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids_1": tokens_1.input_ids.squeeze(0),
            "input_ids_2": tokens_2.input_ids.squeeze(0),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--mixed_precision", type=str, default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha; defaults to rank")
    p.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR gamma for loss weighting")
    return p.parse_args()


def compute_snr_weights(timesteps, noise_scheduler, gamma=5.0):
    """Compute Min-SNR weighting (https://arxiv.org/abs/2303.09556)."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    msnr_weight = torch.clamp(snr, max=gamma) / snr
    return msnr_weight


def main():
    args = parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = args.rank

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    # --- Load models ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float16)
    text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze everything
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Move frozen models to device
    vae.to(accelerator.device)
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    # --- Apply LoRA to UNet ---
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # --- Dataset & DataLoader ---
    dataset = CaptionedImageDataset(args.train_data_dir, args.resolution, tokenizer_1, tokenizer_2)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    num_update_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_update_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # --- Training loop ---
    global_step = 0
    progress_bar = tqdm(range(num_update_steps), desc="Training", disable=not accelerator.is_local_main_process)

    unet.train()
    while global_step < num_update_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=unet.dtype)

                # Encode text
                with torch.no_grad():
                    enc_out_1 = text_encoder_1(batch["input_ids_1"], output_hidden_states=True)
                    enc_out_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)
                    # SDXL uses penultimate hidden states from encoder 1, last from encoder 2
                    text_embeds_1 = enc_out_1.hidden_states[-2]
                    text_embeds_2 = enc_out_2.hidden_states[-2]
                    prompt_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)
                    pooled_prompt_embeds = enc_out_2[0]

                # Sample noise & timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=latents.device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL additional conditioning: original_size, crop, target_size
                add_time_ids = torch.tensor(
                    [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                    dtype=prompt_embeds.dtype, device=latents.device
                ).repeat(latents.shape[0], 1)

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # Predict noise
                noise_pred = unet(
                    noisy_latents, timesteps, prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # Loss with Min-SNR weighting
                if args.snr_gamma > 0:
                    snr_weights = compute_snr_weights(timesteps, noise_scheduler, args.snr_gamma)
                    loss = (snr_weights * F.mse_loss(noise_pred, noise, reduction="none").mean(dim=[1, 2, 3])).mean()
                else:
                    loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                # Checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint at step {global_step}")

                if global_step >= num_update_steps:
                    break

    # --- Save final weights ---
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final"
        accelerator.unwrap_model(unet).save_pretrained(final_dir)
        print(f"Training complete. Final LoRA saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
