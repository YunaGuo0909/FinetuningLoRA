"""LoRA fine-tuning for MDM.

Usage:
    accelerate launch src/training/train_mdm_lora.py \
        --checkpoint_dir /transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512 \
        --style_data_dir /transfer/loradataset/style_bvh/zombie \
        --humanml3d_dir /transfer/loradataset/humanml3d \
        --output_dir /transfer/loraoutputs/models/lora_bvh_zombie_v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.mdm_official import (
    load_official_mdm,
    replace_attention_layers,
    apply_lora,
    motion_to_mdm_input,
    build_y_dict,
)
from src.models.diffusion import GaussianDiffusion
from src.data.humanml_dataset import HumanML3DDataset, StyleMotionDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str,
                   default="/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512")
    p.add_argument("--humanml3d_dir", type=str, default="/transfer/loradataset/humanml3d")
    p.add_argument("--style_data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine")
    p.add_argument("--foot_vel_weight", type=float, default=0.0)
    p.add_argument("--root_stable_weight", type=float, default=0.0)
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


def collate_fn(batch):
    return {
        "motion": torch.stack([b["motion"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "length": [b["length"] for b in batch],
    }


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_official_mdm(args.checkpoint_dir, device="cpu")
    model = replace_attention_layers(model)
    model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)

    diffusion = GaussianDiffusion(args.num_timesteps, args.beta_schedule)

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

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    progress = tqdm(range(args.max_train_steps), desc=output_dir.name,
                    disable=not accelerator.is_local_main_process)

    model.train()
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                motion = batch["motion"]
                captions = batch["caption"]
                lengths = batch["length"]

                x_0 = motion_to_mdm_input(motion)
                t = torch.randint(0, diffusion.num_timesteps, (motion.shape[0],), device=motion.device)
                x_t = diffusion.q_sample(x_0, t, torch.randn_like(x_0))
                y = build_y_dict(captions, lengths, args.max_motion_length, motion.device)

                # MDM predicts x_0 directly (predict_xstart=True)
                pred_x0 = model(x_t, t, y)

                valid_mask = y["mask"].float()
                loss = ((pred_x0 - x_0) ** 2 * valid_mask).sum() / valid_mask.sum().clamp(min=1) / x_0.shape[1]

                if args.foot_vel_weight > 0 or args.root_stable_weight > 0:
                    pred_motion = pred_x0.squeeze(2).permute(0, 2, 1)  # (B, T, 263)
                    gt_motion = x_0.squeeze(2).permute(0, 2, 1)
                    gt_contact = gt_motion[:, :, 259:263]
                    l_contact = (gt_contact[:, :, 0] + gt_contact[:, :, 1]) / 2
                    r_contact = (gt_contact[:, :, 2] + gt_contact[:, :, 3]) / 2

                if args.foot_vel_weight > 0:
                    pred_vel = pred_motion[:, :, 193:259]  # joint velocities
                    l_mask = (l_contact > 0.5).float().unsqueeze(-1)
                    r_mask = (r_contact > 0.5).float().unsqueeze(-1)
                    # L_Ankle=7, R_Ankle=8, L_Foot=10, R_Foot=11
                    foot_loss = (
                        (pred_vel[:, :, 21:24] ** 2 * l_mask).mean() +  # L_Ankle
                        (pred_vel[:, :, 30:33] ** 2 * l_mask).mean() +  # L_Foot
                        (pred_vel[:, :, 24:27] ** 2 * r_mask).mean() +  # R_Ankle
                        (pred_vel[:, :, 33:36] ** 2 * r_mask).mean()    # R_Foot
                    )
                    loss = loss + args.foot_vel_weight * foot_loss

                if args.root_stable_weight > 0:
                    dual_mask = ((l_contact > 0.5) & (r_contact > 0.5)).float()
                    root_loss = (
                        (pred_motion[:, :, 1] ** 2 * dual_mask).mean() +
                        (pred_motion[:, :, 2] ** 2 * dual_mask).mean()
                    )
                    loss = loss + args.root_stable_weight * root_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(ckpt_dir)

                if global_step >= args.max_train_steps:
                    break

    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        print(f"Saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
