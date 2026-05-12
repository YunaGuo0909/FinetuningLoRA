"""MDM wrapper for LoRA injection. Requires /transfer/mdm_official."""

from __future__ import annotations

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from argparse import Namespace


MDM_REPO_PATH = "/transfer/mdm_official"
if MDM_REPO_PATH not in sys.path:
    sys.path.insert(0, MDM_REPO_PATH)


class SplitQKVAttention(nn.Module):
    """Drop-in for nn.MultiheadAttention with split Q/K/V linears (PEFT-compatible)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = False  # match nn.MultiheadAttention interface
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
                need_weights=False, attn_mask=None, **kwargs):
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


def load_official_mdm(checkpoint_dir: str, device: str = "cpu") -> nn.Module:
    """Load official MDM from checkpoint_dir (must contain model*.pt and args.json)."""
    ckpt_dir = Path(checkpoint_dir)

    # Fill defaults for fields absent in older checkpoints
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

    pt_files = sorted(ckpt_dir.glob("model*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No model*.pt found in {ckpt_dir}")
    ckpt_path = pt_files[-1]

    from model.mdm import MDM

    njoints = 263 if args.dataset == "humanml" else 251
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

    # Skip SMPL init — only needed for rot2xyz viz, not for hml_vec inference
    import model.smpl as _smpl_module
    _orig_smpl_init = _smpl_module.SMPL.__init__
    def _dummy_smpl_init(self, **kwargs):
        nn.Module.__init__(self)
    _smpl_module.SMPL.__init__ = _dummy_smpl_init

    model = MDM(**model_args)
    _smpl_module.SMPL.__init__ = _orig_smpl_init

    state_dict = torch.load(str(ckpt_path), map_location=device)
    # Drop positional encoder and CLIP weights (loaded separately)
    for k in [k for k in state_dict if "sequence_pos_encoder" in k or "clip_model" in k]:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def replace_attention_layers(model: nn.Module) -> nn.Module:
    """Replace MDM's fused MultiheadAttention with SplitQKVAttention. Skips CLIP layers."""
    for name, module in model.named_modules():
        if "clip_model" in name:
            continue
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(module, attr_name, SplitQKVAttention.from_multihead_attention(child))
    return model


def apply_lora(model: nn.Module, rank: int = 16, alpha: int = 16,
               target_modules: list[str] = None, dropout: float = 0.05) -> nn.Module:
    """Inject LoRA adapters into attention projections."""
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
    model = load_official_mdm(checkpoint_dir, device=device)
    model = replace_attention_layers(model)
    return apply_lora(model, rank=rank, alpha=alpha)



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
