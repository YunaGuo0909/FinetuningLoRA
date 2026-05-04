"""LoRA fine-tuning script for MDM (Motion Diffusion Model).

Two modes:
    1. Full pre-training on HumanML3D (optional, if no pretrained weights)
    2. LoRA fine-tuning on style-specific data (main use case)

Usage:
    # LoRA fine-tuning on style data
    accelerate launch src/training/train_mdm_lora.py \
        --pretrained_path pretrained/mdm_humanml3d.pt \
        --style_data_dir data/100STYLE_converted \
        --humanml3d_dir data/HumanML3D \
        --output_dir outputs/models/zombie_walk \
        --lora_rank 16 --max_train_steps 2000

    # Full training from scratch (if needed)
    accelerate launch src/training/train_mdm_lora.py \
        --humanml3d_dir data/HumanML3D \
        --output_dir outputs/models/mdm_base \
        --mode pretrain --max_train_steps 50000
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import open_clip
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.mdm import MDM, load_pretrained_mdm
from src.models.diffusion import GaussianDiffusion
from src.data.humanml_dataset import HumanML3DDataset, StyleMotionDataset


# ---------------------------------------------------------------------------
# CLIP text encoder wrapper
# ---------------------------------------------------------------------------

class CLIPTextEncoder:
    """Frozen CLIP text encoder for motion conditioning."""

    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        return emb.float()  # (B, 512)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MDM LoRA Fine-tuning")
    # Data
    p.add_argument("--humanml3d_dir", type=str, default="/transfer/datasets/HumanML3D")
    p.add_argument("--style_data_dir", type=str, default=None, help="Style dataset for LoRA tuning")
    p.add_argument("--pretrained_path", type=str, default="/transfer/pretrained/mdm_humanml3d.pt",
                   help="Path to pretrained MDM weights")

    # Mode
    p.add_argument("--mode", type=str, default="lora", choices=["pretrain", "lora"])

    # Model
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--nfeats", type=int, default=263)
    p.add_argument("--max_motion_length", type=int, default=196)
    p.add_argument("--clip_model", type=str, default="ViT-B-32")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="to_q,to_k,to_v,to_out")

    # Diffusion
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine")
    p.add_argument("--snr_gamma", type=float, default=5.0)

    # Training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--mixed_precision", type=str, default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpointing_steps", type=int, default=500)

    args = p.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    return args


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def apply_lora(model: MDM, rank: int, alpha: int, target_modules: list[str], dropout: float):
    """Apply LoRA adapters to MDM's attention layers via PEFT."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def collate_fn(batch):
    """Custom collate that handles string captions."""
    motions = torch.stack([b["motion"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    captions = [b["caption"] for b in batch]
    lengths = [b["length"] for b in batch]
    return {"motion": motions, "mask": masks, "caption": captions, "length": lengths}


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build model ---
    model = MDM(
        nfeats=args.nfeats,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        clip_dim=512,
        cond_mode="text",
        max_seq_len=args.max_motion_length,
    )

    # Load pretrained weights
    if args.pretrained_path and Path(args.pretrained_path).exists():
        load_pretrained_mdm(model, args.pretrained_path)
        print(f"Loaded pretrained MDM from {args.pretrained_path}")

    # --- Apply LoRA (fine-tuning mode) ---
    if args.mode == "lora":
        target_modules = [t.strip() for t in args.lora_targets.split(",")]
        model = apply_lora(model, args.lora_rank, args.lora_alpha, target_modules, args.lora_dropout)

    # --- Diffusion ---
    diffusion = GaussianDiffusion(args.num_timesteps, args.beta_schedule)

    # --- CLIP text encoder (frozen) ---
    clip_encoder = CLIPTextEncoder(args.clip_model, device=accelerator.device)

    # --- Dataset ---
    if args.mode == "lora" and args.style_data_dir:
        mean = np.load(Path(args.humanml3d_dir) / "Mean.npy")
        std = np.load(Path(args.humanml3d_dir) / "Std.npy")
        dataset = StyleMotionDataset(
            args.style_data_dir, mean, std,
            max_motion_length=args.max_motion_length,
            nfeats=args.nfeats,
        )
    else:
        dataset = HumanML3DDataset(
            args.humanml3d_dir, split="train",
            max_motion_length=args.max_motion_length,
            nfeats=args.nfeats,
        )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # --- Optimizer & Scheduler ---
    from diffusers.optimization import get_scheduler

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=1e-2,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerate
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler,
    )

    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Training loop ---
    global_step = 0
    progress = tqdm(range(args.max_train_steps), desc="Training", disable=not accelerator.is_local_main_process)

    model.train()
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                motion = batch["motion"]  # (B, T, 263)
                mask = batch["mask"]      # (B, T)
                captions = batch["caption"]

                # Encode text
                with torch.no_grad():
                    text_emb = clip_encoder.encode(captions)  # (B, 512)

                # Sample timesteps
                t = torch.randint(0, diffusion.num_timesteps, (motion.shape[0],), device=motion.device)

                # Compute loss
                loss = diffusion.training_losses(
                    model, motion, t, text_emb=text_emb, mask=mask, snr_gamma=args.snr_gamma,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                # Checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    if args.mode == "lora":
                        unwrapped.save_pretrained(ckpt_dir)
                    else:
                        torch.save(unwrapped.state_dict(), ckpt_dir / "model.pt")
                    print(f"\nCheckpoint saved: step {global_step}")

                if global_step >= args.max_train_steps:
                    break

    # --- Save final ---
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        if args.mode == "lora":
            unwrapped.save_pretrained(final_dir)
        else:
            torch.save(unwrapped.state_dict(), final_dir / "model.pt")
        print(f"Training complete. Saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
