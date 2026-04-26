"""Discrete diffusion Transformer for audio-conditioned synth token sequences."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.position_encoding import PositionalEncoding1D, PositionEmbeddingSine
from src.models.components.transformer_layers import Transformer
from src.models.components.transformer_non_autoregressive import CNNBackbone
from src.models.synth_token_space import SynthTokenSpace
from src.utils.probability import GaussianKernelConv


class DiscreteDiffusionParamTransformer(nn.Module):
    """Bidirectional masked-token Transformer over a frozen synth token space."""

    def __init__(
        self,
        *,
        token_space: SynthTokenSpace,
        diffusion_T: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = True,
        gaussian_sigma: float = 0.02,
        gaussian_smoothing_positions: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.token_space = token_space
        self.order = list(token_space.order)
        self.cardinals = list(token_space.cardinalities)
        if int(diffusion_T) != int(len(self.order)):
            raise ValueError(
                "Discrete diffusion requires T == sequence length L. "
                f"Got T={int(diffusion_T)} but L={len(self.order)}."
            )

        self.seq_len = len(self.cardinals)
        self.max_card = max(self.cardinals)
        self.mask_token_id = int(self.max_card)

        self.backbone = CNNBackbone(in_channels=1, d_model=d_model)
        self.enc_pos_embed = PositionEmbeddingSine(d_model // 2)
        self.query_pos_embed = PositionalEncoding1D(d_model, self.seq_len)
        self.transformer = Transformer(
            num_queries=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

        self.token_embed = nn.Embedding(self.max_card + 1, d_model)
        self.time_embed = nn.Embedding(int(diffusion_T) + 1, d_model)
        self.proj = nn.Linear(d_model, self.max_card)
        self.dropout = nn.Dropout(float(dropout))
        self.gaussian_conv = GaussianKernelConv(sigma=gaussian_sigma)
        self.gaussian_smoothing_positions = {int(i) for i in gaussian_smoothing_positions}

        cardinals_t = torch.tensor(self.cardinals, dtype=torch.long)
        class_ids = torch.arange(self.max_card, dtype=torch.long)[None, :].expand(self.seq_len, -1)
        invalid = class_ids >= cardinals_t[:, None]
        self.register_buffer("invalid_class_mask", invalid, persistent=False)

    def forward(self, spectrogram: torch.Tensor, token_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.seq_len:
            raise ValueError(f"Expected token_ids shape (B,{self.seq_len}), got {tuple(token_ids.shape)}")
        if t.ndim != 1 or t.shape[0] != token_ids.shape[0]:
            raise ValueError(f"Expected t shape (B,), got {tuple(t.shape)} for batch {token_ids.shape[0]}")

        batch = token_ids.size(0)
        feats = self.backbone(spectrogram)
        pos_enc = self.enc_pos_embed(feats)
        src = feats.flatten(2).permute(2, 0, 1)
        src_pos = pos_enc.flatten(2).permute(2, 0, 1)
        memory = self.transformer.encoder(src, pos=src_pos)

        tok = self.token_embed(token_ids)
        time = self.time_embed(t.to(dtype=torch.long)).unsqueeze(1)
        tgt = (tok + time).transpose(0, 1)
        query_pos = (
            self.query_pos_embed.encoding[0, : self.seq_len, :]
            .unsqueeze(1)
            .repeat(1, batch, 1)
            .to(tgt.device)
        )

        dec_out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=None,
            pos=src_pos,
            query_pos=query_pos,
        )
        if dec_out.dim() == 4:
            dec_out = dec_out[-1]
        dec_out = dec_out.transpose(0, 1)
        logits = self.proj(self.dropout(dec_out))
        invalid = self.invalid_class_mask.to(device=logits.device)
        return logits.masked_fill(invalid.unsqueeze(0), float("-inf"))

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        midi_label_smoothing: float = 0.0,
        midi_label_smoothing_apply: Sequence[bool] | None = None,
        token_weights: torch.Tensor | None = None,
        uncertainty_weighting: bool = False,
        return_components: bool = False,
        return_midi_per_head: bool = False,
        return_per_param_loss: bool = False,
    ):
        device = logits.device
        if token_weights is not None:
            if not isinstance(token_weights, torch.Tensor):
                token_weights = torch.as_tensor(token_weights, dtype=logits.dtype, device=device)
            else:
                token_weights = token_weights.to(device=device, dtype=logits.dtype)
            if token_weights.ndim != 1 or int(token_weights.numel()) != int(self.seq_len):
                raise ValueError(
                    "token_weights must be a 1D tensor with length equal to the token sequence length "
                    f"({self.seq_len}), got shape {tuple(token_weights.shape)}"
                )
            if (token_weights < 0).any():
                raise ValueError("token_weights must be non-negative.")

        if targets.shape[:2] != logits.shape[:2]:
            raise ValueError(f"targets shape {tuple(targets.shape)} must match logits {tuple(logits.shape[:2])}")
        if loss_mask.shape != targets.shape:
            raise ValueError(f"loss_mask shape {tuple(loss_mask.shape)} must match targets {tuple(targets.shape)}")

        total_loss_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        total_weight_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        param_loss_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        param_weight_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        midi_loss_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        midi_weight_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        midi_per_head_losses = []
        per_token_loss_stats = []

        midi_apply = list(midi_label_smoothing_apply) if midi_label_smoothing_apply is not None else []
        midi_head = 0

        for i, cardinal in enumerate(self.cardinals):
            if cardinal <= 1:
                continue
            mask_i = loss_mask[:, i]
            if not torch.any(mask_i):
                if self.token_space.fields[i].is_midi:
                    midi_head += 1
                continue

            step_logits = logits[:, i, :cardinal][mask_i]
            step_targets = targets[:, i][mask_i]
            target_probs = F.one_hot(step_targets, num_classes=cardinal).float()

            if self.token_space.fields[i].is_midi:
                if midi_label_smoothing > 0:
                    apply_gaussian = midi_head < len(midi_apply) and bool(midi_apply[midi_head])
                    if apply_gaussian:
                        target_probs = torch.zeros_like(target_probs)
                        target_probs.scatter_(1, step_targets.unsqueeze(1), 1.0)
                        target_probs = self.gaussian_conv(target_probs)
                        target_probs = F.normalize(target_probs, p=1, dim=-1)
                    else:
                        smooth = midi_label_smoothing / float(max(cardinal - 1, 1))
                        target_probs = torch.full_like(target_probs, smooth)
                        target_probs.scatter_(1, step_targets.unsqueeze(1), 1.0 - midi_label_smoothing)
                midi_head += 1
            elif i in self.gaussian_smoothing_positions and label_smoothing > 0:
                target_probs = self.gaussian_conv(target_probs)

            log_probs = F.log_softmax(step_logits / float(temperature), dim=-1)
            if target_probs.shape != log_probs.shape:
                raise RuntimeError(
                    f"Shape mismatch for token '{self.order[i]}' (index {i}, cardinal {cardinal}): "
                    f"target_probs {target_probs.shape}, log_probs {log_probs.shape}"
                )

            loss_vec = -(target_probs * log_probs).sum(dim=-1)
            n = loss_vec.numel()
            w = token_weights[i] if token_weights is not None else torch.tensor(1.0, device=device, dtype=logits.dtype)

            loss_mean = loss_vec.mean()
            total_loss_sum = total_loss_sum + w * loss_vec.sum()
            total_weight_sum = total_weight_sum + w * float(n)

            if self.token_space.fields[i].is_midi:
                midi_loss_sum = midi_loss_sum + w * loss_vec.sum()
                midi_weight_sum = midi_weight_sum + w * float(n)
                if return_midi_per_head:
                    midi_per_head_losses.append(loss_mean)
            else:
                param_loss_sum = param_loss_sum + w * loss_vec.sum()
                param_weight_sum = param_weight_sum + w * float(n)

            if return_per_param_loss:
                per_token_loss_stats.append((self.order[i], loss_mean, int(n)))

        if uncertainty_weighting:
            raise ValueError("uncertainty_weighting is not supported in discrete diffusion loss.")

        if total_weight_sum.item() == 0.0:
            total_loss = logits[:, :, 0].sum() * 0.0
        else:
            total_loss = total_loss_sum / (total_weight_sum + 1e-12)

        param_loss = (
            torch.tensor(0.0, device=device, dtype=logits.dtype)
            if param_weight_sum.item() == 0.0
            else (param_loss_sum / (param_weight_sum + 1e-12))
        )
        midi_loss = (
            torch.tensor(0.0, device=device, dtype=logits.dtype)
            if midi_weight_sum.item() == 0.0
            else (midi_loss_sum / (midi_weight_sum + 1e-12))
        )

        if return_components and return_midi_per_head and return_per_param_loss:
            return (
                total_loss,
                param_loss,
                midi_loss,
                torch.stack(midi_per_head_losses) if midi_per_head_losses else torch.tensor([], device=device),
                per_token_loss_stats,
            )
        if return_components and return_midi_per_head:
            return (
                total_loss,
                param_loss,
                midi_loss,
                torch.stack(midi_per_head_losses) if midi_per_head_losses else torch.tensor([], device=device),
            )
        if return_components and return_per_param_loss:
            return total_loss, param_loss, midi_loss, per_token_loss_stats
        if return_components:
            return total_loss, param_loss, midi_loss
        return total_loss
