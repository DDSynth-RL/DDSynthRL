"""Flow-matching encoder and vector-field components (AST + ApproxEquivTransformer).

This file is adapted from synth-permutations' `src/models/components/transformer.py`,
trimmed to the parts needed for Dexed flow matching in AR-Matching.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(
        self, size: int, num_pos: int, init: Literal["zeros", "norm0.02"] = "zeros"
    ):
        super().__init__()
        if init == "zeros":
            pe = torch.zeros(1, num_pos, size)
        else:
            pe = torch.randn(1, num_pos, size) * 0.02
        self.pe = nn.Parameter(pe)

    def penalty(self) -> torch.Tensor:
        return self.pe.norm(2.0, dim=-1).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, : x.shape[1], :]
        return x + pe


class LearntProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_token: int,
        num_params: int,
        num_tokens: int,
        initial_ffn: bool = True,
        final_ffn: bool = True,
    ):
        super().__init__()

        assignment = torch.full(
            (num_tokens, num_params), 1.0 / math.sqrt(num_tokens * num_params)
        )
        assignment = assignment + 1e-4 * torch.randn_like(assignment)
        self._assignment = nn.Parameter(assignment)

        proj = torch.randn(1, d_token) / math.sqrt(d_token)
        proj = proj.repeat(num_params, 1)
        proj = proj + 1e-4 * torch.randn_like(proj)
        self._in_projection = nn.Parameter(proj.clone())
        self._out_projection = nn.Parameter(proj.T.clone())

        if initial_ffn:
            self.initial_ffn = nn.Sequential(
                nn.Linear(d_token, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.initial_ffn = None

        if final_ffn:
            self.final_ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_token),
            )
        elif d_token == d_model:
            self.final_ffn = None
        else:
            self.final_ffn = nn.Linear(d_model, d_token)

    @property
    def assignment(self):
        return self._assignment

    @property
    def in_projection(self):
        return self._in_projection

    @property
    def out_projection(self):
        return self._out_projection

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("bn,nd->bnd", x, self.in_projection)
        if self.initial_ffn is not None:
            values = self.initial_ffn(values)
        tokens = torch.einsum("bnd,kn->bkd", values, self.assignment)
        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)
        if self.final_ffn is not None:
            deassigned = self.final_ffn(deassigned)
        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        return self.assignment.abs().mean()


class DiTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        conditioning_dim: int,
        num_heads: int,
        d_ff: int,
        norm: Literal["layer", "rms"] = "layer",
        first_norm: bool = True,
        adaln_mode: Literal["basic", "zero", "res"] = "basic",
        zero_init: bool = True,
    ):
        super().__init__()
        if first_norm:
            self.norm1 = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
        else:
            self.norm1 = nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        cond_out_dim = d_model * 6 if adaln_mode != "res" else d_model * 4
        self.adaln_mode = adaln_mode
        self.cond = nn.Sequential(
            nn.GELU(),
            nn.Linear(conditioning_dim, cond_out_dim),
        )
        self._init_adaln(adaln_mode)
        self._init_ffn(zero_init)
        self._init_attn(zero_init)

    def _init_adaln(self, mode: Literal["basic", "zero"]):
        if mode == "zero":
            nn.init.constant_(self.cond[-1].weight, 0.0)
            nn.init.constant_(self.cond[-1].bias, 0.0)

    def _init_ffn(self, zero_init: bool):
        nn.init.xavier_normal_(self.ff[0].weight)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.zeros_(self.ff[-1].bias)
        if zero_init:
            nn.init.zeros_(self.ff[-1].weight)
        else:
            nn.init.xavier_normal_(self.ff[-1].weight)

    def _init_attn(self, zero_init: bool):
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.adaln_mode == "res":
            g1, b1, g2, b2 = self.cond(z)[:, None].chunk(4, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = self.cond(z)[:, None].chunk(6, dim=-1)

        res = x
        x = self.norm1(x)
        x = g1 * x + b1
        x = self.attn(x, x, x)[0]
        x = x + res if self.adaln_mode == "res" else a1 * x + res

        res = x
        x = self.norm2(x)
        x = g2 * x + b2
        x = self.ff(x)
        x = x + res if self.adaln_mode == "res" else a2 * x + res
        return x


class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        k = torch.arange(0, half)
        basis = 1 / torch.pow(10000, k / half)
        self.register_buffer("basis", basis[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 1:
            basis = self.basis
        else:
            basis = self.basis[None, :]
            x = x[:, :, None]
        cos_part = torch.cos(x * self.basis)
        sin_part = torch.sin(x * self.basis)
        return torch.cat([cos_part, sin_part], dim=-1)


class ApproxEquivTransformer(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        num_layers: int = 5,
        d_model: int = 1024,
        conditioning_dim: int = 128,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_tokens: int = 32,
        learn_pe: bool = False,
        learn_projection: bool = False,
        pe_type: Literal["initial", "layerwise", "none"] = "initial",
        pe_penalty: float = 0.0,
        time_encoding: Literal["sinusoidal", "scalar"] = "scalar",
        d_enc: int = 256,
        projection_penalty: float = 0.0,
        norm: Literal["layer", "rms"] = "layer",
        skip_first_norm: bool = False,
        adaln_mode: Literal["basic", "zero"] = "basic",
        zero_init: bool = True,
        outer_residual: bool = False,
    ):
        super().__init__()

        self.cfg_dropout_token = nn.Parameter(torch.randn(1, conditioning_dim))
        conditioning_dim = conditioning_dim + (1 if time_encoding == "scalar" else d_enc)
        self.conditioning_ffn = nn.Sequential(
            nn.Linear(conditioning_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList(
            [
                DiTransformerBlock(
                    d_model,
                    d_model,
                    num_heads,
                    d_ff,
                    norm,
                    first_norm=False if i == 0 and skip_first_norm else True,
                    adaln_mode=adaln_mode,
                    zero_init=zero_init,
                )
                for i in range(num_layers)
            ]
        )

        if time_encoding == "sinusoidal":
            self.time_encoding = SinusoidalEncoding(d_enc)
        elif time_encoding == "scalar":
            self.time_encoding = nn.Identity()
        else:
            raise ValueError("time_encoding must be 'sinusoidal' or 'scalar'")

        if pe_type == "initial":
            self.pe = PositionalEncoding(d_model, num_tokens)
            if not learn_pe:
                self.pe.pe.requires_grad = False
        elif pe_type == "layerwise":
            self.pe = nn.ModuleList([PositionalEncoding(d_model, num_tokens) for _ in range(num_layers)])
            if not learn_pe:
                for pe in self.pe:
                    pe.pe.requires_grad = False
        elif pe_type == "none":
            self.pe = None
        else:
            raise ValueError("pe_type must be one of 'initial', 'layerwise', 'none'")

        self.pe_type = pe_type
        self.projection = projection
        if not learn_projection and hasattr(self.projection, "proj"):
            self.projection.proj.requires_grad = False

        self.pe_penalty = pe_penalty
        self.projection_penalty = projection_penalty
        self.outer_residual = outer_residual

    def apply_dropout(self, z: torch.Tensor, rate: float = 0.1):
        if rate == 0.0:
            return z
        dropout_mask = torch.rand(z.shape[0], 1, device=z.device) > rate
        if z.ndim == 3:
            dropout_mask = dropout_mask.unsqueeze(-1)
        return z.where(dropout_mask, self.cfg_dropout_token)

    def penalty(self) -> torch.Tensor:
        penalty = 0.0
        if self.pe_type != "none" and self.pe_penalty > 0.0:
            if self.pe_type == "initial":
                pe_penalty = self.pe.penalty()
            else:
                pe_penalty = sum(pe.penalty() for pe in self.pe)
            penalty += pe_penalty * self.pe_penalty
        if self.projection_penalty > 0.0:
            penalty += self.projection.penalty() * self.projection_penalty
        return penalty

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if conditioning is None:
            conditioning = self.cfg_dropout_token.expand(x.shape[0], -1)

        outer_residual = x if self.outer_residual else None
        x = self.projection.param_to_token(x)
        t = self.time_encoding(t)

        layerwise_conditioning = False
        if conditioning.ndim == 3:
            t = t.unsqueeze(1).repeat(1, conditioning.shape[1], 1)
            layerwise_conditioning = True

        z = torch.cat((conditioning, t), dim=-1)
        z = self.conditioning_ffn(z)

        if self.pe_type == "initial":
            x = self.pe(x)

        for i, layer in enumerate(self.layers):
            if self.pe_type == "layerwise":
                x = self.pe[i](x)
            z_ = z[:, i, :] if layerwise_conditioning else z
            x = layer(x, z_)

        x = self.projection.token_to_param(x)
        if outer_residual is not None:
            x = x + outer_residual
        return x


class PatchEmbed(nn.Module):
    """Convolutional patch encoder like in AST/ViT, with overlap."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        d_model: int,
        spec_shape: Tuple[int, int] = (128, 401),
    ):
        super().__init__()
        assert stride < patch_size, "Overlap must be less than patch size"

        mel_padding = (stride - (spec_shape[0] - patch_size)) % stride
        time_padding = (stride - (spec_shape[1] - patch_size)) % stride

        self.pad = nn.ZeroPad2d((0, mel_padding, 0, time_padding))
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )
        self.num_tokens = self._get_num_tokens(in_channels, spec_shape)

    def _get_num_tokens(self, in_channels: int, spec_shape: Tuple[int, int]) -> int:
        x = torch.randn(1, in_channels, *spec_shape, device=self.projection.weight.device)
        out_shape = self.projection(self.pad(x)).shape
        return math.prod(out_shape[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AudioSpectrogramTransformer(nn.Module):
    """AST-style encoder producing multiple conditioning tokens."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 16,
        n_conditioning_outputs: int = 12,
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 1,
        spec_shape: Tuple[int, int] = (128, 401),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            stride=patch_stride,
            in_channels=input_channels,
            d_model=d_model,
            spec_shape=spec_shape,
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            self.patch_embed.num_tokens + n_conditioning_outputs,
            init="norm0.02",
        )
        self.embed_tokens = nn.Parameter(
            torch.empty(1, n_conditioning_outputs, d_model).normal_(0.0, 1e-6)
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_model,
                    0.0,
                    "gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        embed_tokens = self.embed_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((embed_tokens, x), dim=1)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = x[:, : self.embed_tokens.shape[1]]
        x = self.out_proj(x)
        return x

