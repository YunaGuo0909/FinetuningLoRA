"""Motion Diffusion Model (MDM) with LoRA-friendly separate Q/K/V attention.

Architecture follows Tevet et al. (2023) "Human Motion Diffusion Model" but
replaces nn.MultiheadAttention with explicit Linear projections so that PEFT
LoRA can target to_q / to_k / to_v / to_out directly.

Pretrained weights from the official MDM repo can be loaded via
`load_pretrained_mdm()` which splits the fused in_proj_weight automatically.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = math.log(10_000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -freq)
        emb = t[:, None].float() * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(timesteps)


class MultiheadSelfAttention(nn.Module):
    """Self-attention with explicit Q/K/V linear layers (LoRA targets)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.to_q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            # key_padding_mask: (B, T), True = ignore
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm style (matches official MDM behavior after conversion)
        h = self.norm1(x)
        h = self.self_attn(h, key_padding_mask)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.linear2(self.dropout2(F.gelu(self.linear1(h))))
        x = x + self.dropout3(h)
        return x


# ---------------------------------------------------------------------------
# MDM
# ---------------------------------------------------------------------------

class MDM(nn.Module):
    """Motion Diffusion Model.

    Input:  noisy motion x_t (B, T, nfeats), timestep, text embedding
    Output: predicted noise eps (B, T, nfeats)
    """

    def __init__(
        self,
        nfeats: int = 263,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        clip_dim: int = 512,
        cond_mode: str = "text",
        max_seq_len: int = 196,
    ):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.cond_mode = cond_mode

        # --- Input / output projections (2-layer MLP, matching official MDM) ---
        self.input_process = nn.Sequential(
            nn.Linear(nfeats, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.output_process = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, nfeats),
        )

        # --- Positional encoding (learnable) ---
        self.pos_embedding = nn.Embedding(max_seq_len + 2, latent_dim)  # +2 for cond tokens

        # --- Condition embeddings ---
        self.embed_timestep = TimestepEmbedder(latent_dim)
        if "text" in cond_mode:
            self.embed_text = nn.Linear(clip_dim, latent_dim)

        # --- Transformer encoder ---
        self.transformer = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, nfeats) noisy motion
            timesteps: (B,) diffusion timestep indices
            text_emb:  (B, clip_dim) CLIP text features
            mask:      (B, T) True where frames are padding
        Returns:
            (B, T, nfeats) predicted noise
        """
        B, T, _ = x.shape

        # Project motion to latent dim
        h = self.input_process(x)  # (B, T, D)

        # Build condition tokens
        cond_tokens = []
        t_emb = self.embed_timestep(timesteps)  # (B, D)
        cond_tokens.append(t_emb.unsqueeze(1))

        if text_emb is not None and hasattr(self, "embed_text"):
            txt = self.embed_text(text_emb)  # (B, D)
            cond_tokens.append(txt.unsqueeze(1))

        n_cond = len(cond_tokens)
        cond = torch.cat(cond_tokens, dim=1)  # (B, n_cond, D)
        h = torch.cat([cond, h], dim=1)  # (B, n_cond + T, D)

        # Add positional encoding
        seq_len = h.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        h = h + self.pos_embedding(positions)

        # Build key_padding_mask for transformer (True = ignore)
        if mask is not None:
            cond_mask = torch.zeros(B, n_cond, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cond_mask, mask], dim=1)
        else:
            key_padding_mask = None

        # Transformer
        for block in self.transformer:
            h = block(h, key_padding_mask)

        h = self.final_norm(h)

        # Strip condition tokens, project back
        h = h[:, n_cond:]  # (B, T, D)
        output = self.output_process(h)  # (B, T, nfeats)
        return output


# ---------------------------------------------------------------------------
# Weight conversion from official MDM checkpoint
# ---------------------------------------------------------------------------

def load_pretrained_mdm(model: MDM, ckpt_path: str, strict: bool = False) -> MDM:
    """Load official MDM checkpoint, remapping fused QKV weights to separate layers.

    Official MDM uses nn.TransformerEncoder with fused in_proj_weight.
    Our model uses separate to_q / to_k / to_v linear layers.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt)

    new_state = {}
    d = model.latent_dim

    for key, val in state.items():
        # --- Input process ---
        if key.startswith("input_process.poseEmbedding."):
            idx = key.split(".")[-2]  # 0 or 1
            suffix = key.split(".")[-1]  # weight or bias
            new_key = f"input_process.{int(idx) * 2}.{suffix}"
            new_state[new_key] = val

        # --- Output process ---
        elif key.startswith("output_process.poseFinal."):
            idx = key.split(".")[-2]
            suffix = key.split(".")[-1]
            new_key = f"output_process.{int(idx) * 2}.{suffix}"
            new_state[new_key] = val

        # --- Timestep embedder ---
        elif key.startswith("embed_timestep.time_embed."):
            new_key = key.replace("embed_timestep.time_embed.", "embed_timestep.time_embed.")
            new_state[new_key] = val

        # --- Text projection ---
        elif key.startswith("embed_text."):
            new_state[key] = val

        # --- Transformer layers ---
        elif "seqTransEncoder.layers." in key:
            layer_idx = key.split(".")[2]
            rest = ".".join(key.split(".")[3:])

            # Split fused attention weights
            if rest == "self_attn.in_proj_weight":
                new_state[f"transformer.{layer_idx}.self_attn.to_q.weight"] = val[:d]
                new_state[f"transformer.{layer_idx}.self_attn.to_k.weight"] = val[d:2*d]
                new_state[f"transformer.{layer_idx}.self_attn.to_v.weight"] = val[2*d:]
            elif rest == "self_attn.in_proj_bias":
                new_state[f"transformer.{layer_idx}.self_attn.to_q.bias"] = val[:d]
                new_state[f"transformer.{layer_idx}.self_attn.to_k.bias"] = val[d:2*d]
                new_state[f"transformer.{layer_idx}.self_attn.to_v.bias"] = val[2*d:]
            elif rest.startswith("self_attn.out_proj."):
                suffix = rest.split(".")[-1]
                new_state[f"transformer.{layer_idx}.self_attn.to_out.{suffix}"] = val
            else:
                # linear1, linear2, norm1, norm2
                new_state[f"transformer.{layer_idx}.{rest}"] = val

        # Skip positional encoding buffers (we use learned embeddings instead)

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        print(f"Missing keys (expected for pos_embedding etc.): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    print(f"Loaded pretrained MDM from {ckpt_path}")
    return model
