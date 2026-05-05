"""Gaussian Diffusion process for motion generation.

Supports DDPM training and both DDPM / DDIM sampling.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import numpy as np


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
    betas = 1 - f[1:] / f[:-1]
    return betas.clamp(max=0.999).float()


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)


class GaussianDiffusion:
    """Gaussian diffusion process for training and sampling."""

    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "cosine"):
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)

        self.num_timesteps = num_timesteps
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod_minus_one = (1.0 / self.alphas_cumprod - 1).sqrt()

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * alphas.sqrt() / (1.0 - self.alphas_cumprod)

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Gather schedule values at timestep t and reshape for broadcasting."""
        out = schedule.to(t.device).gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    # --- Forward process ---

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Add noise to x_0 at timestep t: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    # --- Training loss ---

    def training_losses(self, model, x_0, t, text_emb=None, mask=None, snr_gamma=0.0):
        """Compute diffusion training loss (noise prediction).

        Args:
            model: denoising network (MDM)
            x_0: clean motion (B, T, D)
            t: timestep indices (B,)
            text_emb: CLIP text embeddings (B, clip_dim)
            mask: padding mask (B, T)
            snr_gamma: Min-SNR gamma weighting (0 = disabled)
        Returns:
            loss scalar
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Mask padded frames in input
        if mask is not None:
            x_t = x_t * (~mask).unsqueeze(-1).float()

        pred_noise = model(x_t, t, text_emb=text_emb, mask=mask)

        if mask is not None:
            # Only compute loss on valid frames
            valid = (~mask).unsqueeze(-1).float()
            per_frame_loss = ((pred_noise - noise) ** 2 * valid).sum(dim=-1).mean(dim=-1)
            valid_frames = valid.squeeze(-1).sum(dim=-1).clamp(min=1)
            per_sample_loss = per_frame_loss / valid_frames
        else:
            per_sample_loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=[1, 2])

        if snr_gamma > 0:
            snr = self._compute_snr(t)
            weight = torch.clamp(snr, max=snr_gamma) / snr
            per_sample_loss = weight * per_sample_loss

        return per_sample_loss.mean()

    def _compute_snr(self, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod.to(t.device)[t]
        return alpha_bar / (1 - alpha_bar)

    # --- DDPM sampling ---

    def _predict_x0_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recip_alphas_cumprod_minus_one, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, model, x_t, t, text_emb=None, mask=None):
        """Single DDPM reverse step."""
        pred_noise = model(x_t, t, text_emb=text_emb, mask=mask)
        x_0_pred = self._predict_x0_from_noise(x_t, t, pred_noise)

        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = self._extract(self.posterior_log_variance, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, text_emb=None, mask=None, device="cuda"):
        """Full DDPM sampling from noise to motion."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, text_emb=text_emb, mask=mask)
            if mask is not None:
                x = x * (~mask).unsqueeze(-1).float()
        return x

    # --- DDIM sampling (faster) ---

    @torch.no_grad()
    def ddim_sample(self, model, shape, text_emb=None, mask=None, device="cuda",
                    num_steps: int = 50, eta: float = 0.0):
        """DDIM accelerated sampling."""
        # Sub-select timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, t, text_emb=text_emb, mask=mask)

            alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = self._extract(
                self.alphas_cumprod,
                torch.full_like(t, timesteps[i + 1] if i + 1 < len(timesteps) else 0),
                x.shape,
            )

            x_0_pred = (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * pred_noise
            noise = torch.randn_like(x) if t_val > 0 else 0
            x = alpha_bar_prev.sqrt() * x_0_pred + dir_xt + sigma * noise

            if mask is not None:
                x = x * (~mask).unsqueeze(-1).float()

        return x
