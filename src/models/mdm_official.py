"""Wrapper around the official MDM codebase for LoRA injection.

Loads the official MDM model with pretrained weights, then replaces
nn.MultiheadAttention with separate Q/K/V layers for PEFT LoRA support.

Requires: /transfer/mdm_official (cloned official MDM repo)
"""

from __future__ import annotations

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from argparse import Namespace


# ---------------------------------------------------------------------------
# Add official MDM repo to path
# ---------------------------------------------------------------------------

MDM_REPO_PATH = "/transfer/mdm_official"
if MDM_REPO_PATH not in sys.path:
    sys.path.insert(0, MDM_REPO_PATH)


# ---------------------------------------------------------------------------
# Drop-in replacement for nn.MultiheadAttention with separate Q/K/V
# ---------------------------------------------------------------------------

class SplitQKVAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with explicit Q/K/V.

    Accepts the same forward signature as nn.MultiheadAttention so it works
    inside nn.TransformerEncoderLayer without changes.
    LoRA targets: to_q, to_k, to_v, to_out
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    @classmethod
    def from_multihead_attention(cls, mha: nn.MultiheadAttention) -> "SplitQKVAttention":
        """Create from existing nn.MultiheadAttention, copying weights."""
        new = cls(mha.embed_dim, mha.num_heads, mha.dropout)
        d = mha.embed_dim
        # Split fused in_proj_weight (3*D, D) -> Q, K, V
        new.to_q.weight.data.copy_(mha.in_proj_weight[:d])
        new.to_k.weight.data.copy_(mha.in_proj_weight[d:2*d])
        new.to_v.weight.data.copy_(mha.in_proj_weight[2*d:])
        new.to_q.bias.data.copy_(mha.in_proj_bias[:d])
        new.to_k.bias.data.copy_(mha.in_proj_bias[d:2*d])
        new.to_v.bias.data.copy_(mha.in_proj_bias[2*d:])
        new.to_out.weight.data.copy_(mha.out_proj.weight)
        new.to_out.bias.data.copy_(mha.out_proj.bias)
        return new

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        """Same signature as nn.MultiheadAttention.forward().

        Args:
            query, key, value: (T, B, D) — seq_first format
        Returns:
            output: (T, B, D)
            attn_weights: (B, T, T) or None
        """
        T, B, D = query.shape

        q = self.to_q(query).view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.to_k(key).view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.to_v(value).view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(2, 0, 1, 3).contiguous().view(T, B, D)
        out = self.to_out(out)

        if need_weights:
            return out, attn.mean(dim=1)  # average over heads
        return out, None


# ---------------------------------------------------------------------------
# Model loading and LoRA injection
# ---------------------------------------------------------------------------

def load_official_mdm(checkpoint_dir: str, device: str = "cpu") -> nn.Module:
    """Load the official MDM model with pretrained weights.

    Args:
        checkpoint_dir: path containing model*.pt and args.json
            e.g. "/transfer/lorapretrain/humanml_trans_enc_512/humanml_trans_enc_512"
        device: device to load to
    Returns:
        MDM model with loaded weights
    """
    ckpt_dir = Path(checkpoint_dir)

    # Load args, fill in defaults for fields added in newer MDM versions
    args_path = ckpt_dir / "args.json"
    with open(args_path) as f:
        args_dict = json.load(f)

    defaults = {
        "unconstrained": False,
        "keyframe": False,
        "target_joint_names": None,
        "pos_embed_max_len": 5000,
        "multi_target_cond": None,
        "clip_version": "ViT-B/32",
        "text_encoder_type": "clip",
        "mask_frames": False,
        "prefix_len": 0,
    }
    for k, v in defaults.items():
        if k not in args_dict:
            args_dict[k] = v

    args = Namespace(**args_dict)

    # Find the checkpoint file
    pt_files = sorted(ckpt_dir.glob("model*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No model*.pt found in {ckpt_dir}")
    ckpt_path = pt_files[-1]  # latest checkpoint

    # Import official MDM
    from model.mdm import MDM

    # Build model args directly from args.json (avoid needing a data object)
    njoints = 263 if args.dataset == "humanml" else 251  # humanml=263, kit=251
    model_args = {
        "modeltype": "",
        "njoints": njoints,
        "nfeats": 1,
        "num_actions": 1,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "latent_dim": args.latent_dim,
        "ff_size": 1024,
        "num_layers": args.layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "activation": "gelu",
        "data_rep": args.data_rep,
        "dataset": args.dataset,
        "clip_dim": 512,
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": args_dict.get("clip_version", "ViT-B/32"),
        "cond_mode": args.cond_mode,
        "cond_mask_prob": args.cond_mask_prob,
        "action_emb": args.action_emb,
        "legacy": args.legacy,
    }

    # Create model
    model = MDM(**model_args)

    # Load weights
    state_dict = torch.load(str(ckpt_path), map_location=device)
    # Remove PE keys (official pattern from load_model_wo_clip)
    keys_to_remove = [k for k in state_dict.keys()
                      if "sequence_pos_encoder" in k or "clip_model" in k]
    for k in keys_to_remove:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded official MDM from {ckpt_path}")
    print(f"  Arch: {args.arch}, Layers: {args.layers}, Latent: {args.latent_dim}")

    return model.to(device)


def replace_attention_layers(model: nn.Module) -> nn.Module:
    """Replace all nn.MultiheadAttention in the model with SplitQKVAttention.

    This enables PEFT LoRA to target to_q, to_k, to_v, to_out.
    """
    count = 0
    # The official MDM uses seqTransEncoder which contains TransformerEncoderLayers
    if hasattr(model, "seqTransEncoder"):
        for layer in model.seqTransEncoder.layers:
            if hasattr(layer, "self_attn") and isinstance(layer.self_attn, nn.MultiheadAttention):
                new_attn = SplitQKVAttention.from_multihead_attention(layer.self_attn)
                layer.self_attn = new_attn
                count += 1

    print(f"Replaced {count} attention layers with SplitQKVAttention")
    return model


def apply_lora(model: nn.Module, rank: int = 16, alpha: int = 16,
               target_modules: list[str] = None, dropout: float = 0.05) -> nn.Module:
    """Apply LoRA adapters to the model's attention layers."""
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def prepare_mdm_for_lora(checkpoint_dir: str, rank: int = 16, alpha: int = 16,
                          device: str = "cpu") -> nn.Module:
    """Full pipeline: load official MDM → replace attention → apply LoRA."""
    model = load_official_mdm(checkpoint_dir, device=device)
    model = replace_attention_layers(model)
    model = apply_lora(model, rank=rank, alpha=alpha)
    return model


# ---------------------------------------------------------------------------
# Helpers for training/inference
# ---------------------------------------------------------------------------

def motion_to_mdm_input(motion: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, 263) motion to official MDM format (B, 263, 1, T)."""
    B, T, D = motion.shape
    return motion.permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, T)


def mdm_output_to_motion(output: torch.Tensor) -> torch.Tensor:
    """Convert official MDM output (B, 263, 1, T) back to (B, T, 263)."""
    return output.squeeze(2).permute(0, 2, 1)  # (B, T, D)


def build_y_dict(captions: list[str], lengths: list[int], max_len: int,
                 device: torch.device) -> dict:
    """Build the y conditioning dict that official MDM expects."""
    B = len(captions)
    # mask: (B, 1, 1, T) — True where valid frames
    mask = torch.zeros(B, 1, 1, max_len, dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        mask[i, :, :, :length] = True
    return {
        "text": captions,
        "mask": mask,
    }
