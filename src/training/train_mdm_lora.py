"""LoRA fine-tuning script for official MDM (Motion Diffusion Model).

Uses the official MDM codebase with LoRA injected into attention layers.

Usage:
    accelerate launch src/training/train_mdm_lora.py \
        --checkpoint_dir /transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512 \
        --style_data_dir /transfer/loradataset/style_converted \
        --humanml3d_dir /transfer/loradataset/humanml3d \
        --output_dir /transfer/loraoutputs/models/style_lora \
        --lora_rank 16 --max_train_steps 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.mdm_official import (
    load_official_mdm,
    replace_attention_layers,
    apply_lora,
    motion_to_mdm_input,
    mdm_output_to_motion,
    build_y_dict,
)
from src.models.diffusion import GaussianDiffusion
from src.data.humanml_dataset import HumanML3DDataset, StyleMotionDataset


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MDM LoRA Fine-tuning (official model)")
    p.add_argument("--checkpoint_dir", type=str,
                   default="/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512")
    p.add_argument("--humanml3d_dir", type=str, default="/transfer/loradataset/humanml3d")
    p.add_argument("--style_data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Diffusion
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine")
    p.add_argument("--snr_gamma", type=float, default=5.0)

    # Auxiliary losses
    p.add_argument("--foot_vel_weight", type=float, default=0.0,
                   help="Weight for foot velocity penalty during contact (0=disabled)")

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--mixed_precision", type=str, default="no")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--max_motion_length", type=int, default=196)
    p.add_argument("--nfeats", type=int, default=263)

    args = p.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    return args


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    motions = torch.stack([b["motion"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    captions = [b["caption"] for b in batch]
    lengths = [b["length"] for b in batch]
    return {"motion": motions, "mask": masks, "caption": captions, "length": lengths}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load official MDM model ---
    print("Loading official MDM model...")
    model = load_official_mdm(args.checkpoint_dir, device="cpu")

    # Replace attention with split Q/K/V, then apply LoRA
    model = replace_attention_layers(model)
    model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # --- Diffusion ---
    diffusion = GaussianDiffusion(args.num_timesteps, args.beta_schedule)

    # --- Dataset ---
    mean = np.load(Path(args.humanml3d_dir) / "Mean.npy")
    std = np.load(Path(args.humanml3d_dir) / "Std.npy")

    if args.style_data_dir:
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

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler,
    )

    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Training loop ---
    global_step = 0
    progress = tqdm(range(args.max_train_steps), desc="Training",
                    disable=not accelerator.is_local_main_process)

    model.train()
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                motion = batch["motion"]        # (B, T, 263) normalized
                mask = batch["mask"]            # (B, T) bool
                captions = batch["caption"]     # list of strings
                lengths = batch["length"]       # list of ints

                # Convert to MDM format: (B, 263, 1, T)
                x_0 = motion_to_mdm_input(motion)

                # Sample timesteps
                t = torch.randint(0, diffusion.num_timesteps, (motion.shape[0],), device=motion.device)

                # Add noise
                noise = torch.randn_like(x_0)
                x_t = diffusion.q_sample(x_0, t, noise)

                # Build conditioning dict
                y = build_y_dict(captions, lengths, args.max_motion_length, motion.device)

                # Forward pass — official MDM predicts x_0 (predict_xstart=True)
                pred_x0 = model(x_t, t, y)

                # Loss: predict x_0, not noise (matching official MDM training)
                valid_mask = y["mask"].float()  # (B, 1, 1, T)
                loss_per_elem = (pred_x0 - x_0) ** 2 * valid_mask
                loss = loss_per_elem.sum() / valid_mask.sum().clamp(min=1) / x_0.shape[1]

                # Foot velocity penalty: penalize foot joint velocity when contact=1
                # pred_x0 shape: (B, 263, 1, T). We work on the (B, T, 263) view.
                if args.foot_vel_weight > 0:
                    pred_motion = pred_x0.squeeze(2).permute(0, 2, 1)  # (B, T, 263)
                    gt_motion = x_0.squeeze(2).permute(0, 2, 1)

                    # Foot contact from GT: dims 259-262 (l_heel, l_toe, r_heel, r_toe)
                    gt_contact = gt_motion[:, :, 259:263]  # (B, T, 4)
                    l_contact = (gt_contact[:, :, 0] + gt_contact[:, :, 1]) / 2  # (B, T)
                    r_contact = (gt_contact[:, :, 2] + gt_contact[:, :, 3]) / 2

                    # Foot joint velocities from predicted x0
                    # Joint velocities: dims 193-258 (22 joints * 3)
                    # L_Ankle=7, R_Ankle=8, L_Foot=10, R_Foot=11
                    pred_vel = pred_motion[:, :, 193:259]  # (B, T, 66)
                    l_ankle_vel = pred_vel[:, :, 7*3:7*3+3]   # (B, T, 3)
                    l_foot_vel  = pred_vel[:, :, 10*3:10*3+3]
                    r_ankle_vel = pred_vel[:, :, 8*3:8*3+3]
                    r_foot_vel  = pred_vel[:, :, 11*3:11*3+3]

                    # When contact > 0.5, velocity should be ~0
                    l_mask = (l_contact > 0.5).float().unsqueeze(-1)  # (B, T, 1)
                    r_mask = (r_contact > 0.5).float().unsqueeze(-1)

                    foot_loss = (
                        (l_ankle_vel ** 2 * l_mask).mean() +
                        (l_foot_vel ** 2 * l_mask).mean() +
                        (r_ankle_vel ** 2 * r_mask).mean() +
                        (r_foot_vel ** 2 * r_mask).mean()
                    )
                    loss = loss + args.foot_vel_weight * foot_loss

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

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(ckpt_dir)
                    print(f"\nCheckpoint saved: step {global_step}")

                if global_step >= args.max_train_steps:
                    break

    # Save final
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir)
        print(f"Training complete. Saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
